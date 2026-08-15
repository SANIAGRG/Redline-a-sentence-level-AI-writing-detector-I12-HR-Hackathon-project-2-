"""Likelihood signals (Module 4, spec 7A): per-sentence measurements
from Qwen2.5-0.5B base + instruct, loaded via `transformers` (not
Ollama -- raw logits are needed, which Ollama doesn't expose). Qwen2.5
shares lineage with no generator family used anywhere in this project
(DAIGT's 15 families or the 3 locally-generated modern models).

**Sentence sampling (deadline scope cut, ADR 0008):** scoring every
sentence in every document was measured at ~22 sec/doc (both models).
An initial attempt to sample 6 sentences/doc (2 early/mid/late) with
per-sentence forward passes actually cost *more* per document (~26
sec/doc) -- per-call overhead dominates over sequence length at this
scale, so splitting one paragraph-level pass into several sentence-level
passes was a net loss. Reduced to 3 sentences/doc (1 early/mid/late)
to directly cut the number of forward-pass calls. Each sampled sentence
is still scored with its real left context (prior sentences in the same
paragraph), not in isolation. Document-level aggregates are therefore
built from a small *sample* of each document's sentences -- stated
explicitly in docs/EVALUATION.md.

Cross-perplexity ratio is a **simplified** Binoculars-style proxy: ratio
of the two models' perplexity on the observed tokens, not the full
distributional cross-entropy the original paper uses. Kept (not
dropped) because it is the ESL-bias mitigation the audit (C5) depends
on -- documented as a simplification, not cut, under the deadline.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from detector.ingest.segmentation import get_nlp, split_paragraphs

_MODELS: dict[str, tuple[PreTrainedTokenizer, PreTrainedModel]] = {}

N_SAMPLED_SENTENCES = 3
N_BUCKETS = 3  # early / middle / late
PER_BUCKET = N_SAMPLED_SENTENCES // N_BUCKETS


def load_model(name: str) -> tuple[PreTrainedTokenizer, PreTrainedModel]:
    if name not in _MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
        model.eval()
        _MODELS[name] = (tok, model)  # type: ignore[assignment]
    return _MODELS[name]


@dataclass(frozen=True)
class SentenceLikelihoodFeatures:
    doc_id: str
    paragraph_index: int
    sentence_index: int
    text: str
    mean_logprob: float
    mean_logrank: float
    lrr: float
    mean_entropy: float
    mean_curvature: float
    logprob_variance: float
    cross_ppl_ratio: float


@dataclass(frozen=True)
class _CandidateSentence:
    paragraph_index: int
    sentence_index: int
    paragraph_text: str
    char_start: int  # start offset of this sentence within paragraph_text
    char_end: int
    text: str


def _all_sentences(text: str) -> list[_CandidateSentence]:
    nlp = get_nlp()
    out = []
    for para_idx, paragraph in enumerate(split_paragraphs(text)):
        spacy_doc = nlp(paragraph)
        for sent_idx, sent in enumerate(spacy_doc.sents):
            out.append(
                _CandidateSentence(
                    paragraph_index=para_idx,
                    sentence_index=sent_idx,
                    paragraph_text=paragraph,
                    char_start=sent.start_char,
                    char_end=sent.end_char,
                    text=sent.text.strip(),
                )
            )
    return out


def sample_sentences(text: str, seed: int) -> list[_CandidateSentence]:
    """Stratified sample: up to PER_BUCKET sentences from each of the
    early/middle/late thirds of the document (by sentence order).
    """
    all_sents = _all_sentences(text)
    if len(all_sents) <= N_SAMPLED_SENTENCES:
        return all_sents

    rng = random.Random(seed)
    bucket_size = len(all_sents) / N_BUCKETS
    selected = []
    for b in range(N_BUCKETS):
        lo = int(b * bucket_size)
        hi = int((b + 1) * bucket_size) if b < N_BUCKETS - 1 else len(all_sents)
        bucket = all_sents[lo:hi]
        n = min(PER_BUCKET, len(bucket))
        selected.extend(rng.sample(bucket, n))
    return selected


@torch.no_grad()
def _score_sentence_in_context(
    context_text: str, target_char_start: int, tok: PreTrainedTokenizer, model: PreTrainedModel
) -> tuple[list[float], list[float], list[float]]:
    """Forward pass over context_text (all sentences up to and including
    the target, from the same paragraph); returns per-token
    logprob/rank/entropy for only the tokens at or after
    target_char_start (token 0 of the whole context is always excluded
    -- no left context).
    """
    enc = tok(
        context_text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=1024
    )
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()
    logits = model(input_ids.unsqueeze(0)).logits[0]

    logprobs: list[float] = []
    ranks: list[float] = []
    entropies: list[float] = []
    for i in range(1, len(input_ids)):
        if offsets[i][0] < target_char_start:
            continue
        step_logits = logits[i - 1]
        log_probs_i = torch.log_softmax(step_logits, dim=-1)
        token_id = input_ids[i].item()
        token_logprob = log_probs_i[token_id].item()
        rank = int((step_logits > step_logits[token_id]).sum().item()) + 1
        probs_i = log_probs_i.exp()
        entropy = -(probs_i * log_probs_i).sum().item()
        logprobs.append(token_logprob)
        ranks.append(float(rank))
        entropies.append(entropy)
    return logprobs, ranks, entropies


def compute_document_likelihood(
    doc_id: str,
    text: str,
    seed: int = 42,
    base_name: str = "Qwen/Qwen2.5-0.5B",
    instruct_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    candidates: list[_CandidateSentence] | None = None,
) -> list[SentenceLikelihoodFeatures]:
    """By default scores a stratified 3-sentence sample (bulk corpus use,
    see module docstring). Pass `candidates=_all_sentences(text)` to
    score every sentence instead -- affordable for one essay at a time
    (the live app), not for scoring hundreds of documents.
    """
    base_tok, base_model = load_model(base_name)
    instruct_tok, instruct_model = load_model(instruct_name)

    results: list[SentenceLikelihoodFeatures] = []
    for cand in candidates if candidates is not None else sample_sentences(text, seed):
        context_text = cand.paragraph_text[: cand.char_end]

        logprobs, ranks, entropies = _score_sentence_in_context(
            context_text, cand.char_start, base_tok, base_model
        )
        if not logprobs:
            continue
        instruct_logprobs, _, _ = _score_sentence_in_context(
            context_text, cand.char_start, instruct_tok, instruct_model
        )

        mean_logprob = sum(logprobs) / len(logprobs)
        mean_logrank = sum(math.log(r) for r in ranks) / len(ranks)
        mean_entropy = sum(entropies) / len(entropies)
        mean_curvature = sum(lp + e for lp, e in zip(logprobs, entropies, strict=True)) / len(logprobs)
        variance = sum((lp - mean_logprob) ** 2 for lp in logprobs) / len(logprobs)
        lrr = mean_logprob / mean_logrank if mean_logrank else 0.0

        if instruct_logprobs:
            mean_instruct_logprob = sum(instruct_logprobs) / len(instruct_logprobs)
            base_ppl = math.exp(-mean_logprob)
            instruct_ppl = math.exp(-mean_instruct_logprob)
            cross_ppl_ratio = base_ppl / instruct_ppl if instruct_ppl else 1.0
        else:
            cross_ppl_ratio = 1.0

        results.append(
            SentenceLikelihoodFeatures(
                doc_id=doc_id,
                paragraph_index=cand.paragraph_index,
                sentence_index=cand.sentence_index,
                text=cand.text,
                mean_logprob=mean_logprob,
                mean_logrank=mean_logrank,
                lrr=lrr,
                mean_entropy=mean_entropy,
                mean_curvature=mean_curvature,
                logprob_variance=variance,
                cross_ppl_ratio=cross_ppl_ratio,
            )
        )
    return results

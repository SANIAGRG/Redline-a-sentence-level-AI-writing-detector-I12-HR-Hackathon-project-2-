"""Stylometric signals (Module 2, spec section 7B): no LM involved, cheap
to compute, and the kind of signal a human reader can argue with.

Computed at document granularity here, over the 4,000-essay baseline --
this frame is what Module 4 z-scores against (7C: "Z-score every feature
against the 4,000-essay human baseline, conditioned on length band").
Per-sentence scoring in Module 4 reuses these same helpers.

Several signals here are necessarily heuristic (list/Oxford-comma
detection via regex, participial-opener detection, a small curated
homophone-error phrase list) rather than a full grammar parse -- documented
as a known limitation in docs/DATA_CARD.md / docs/LIMITATIONS.md, not
hidden.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from spellchecker import SpellChecker

from detector.ingest.segmentation import get_nlp, split_paragraphs

_SPELL = SpellChecker()

DISCOURSE_MARKERS = frozenset(
    {
        "moreover",
        "furthermore",
        "additionally",
        "consequently",
        "nevertheless",
        "nonetheless",
        "therefore",
        "thus",
        "hence",
        "accordingly",
        "ultimately",
        "crucially",
        "importantly",
        "significantly",
        "notably",
        "specifically",
        "essentially",
        "fundamentally",
        "undoubtedly",
        "clearly",
        "indeed",
        "however",
        "meanwhile",
        "subsequently",
    }
)

# Reliable, well-documented native-English error patterns. Not a general
# homophone/grammar checker -- see module docstring.
_HOMOPHONE_ERROR_PATTERNS = [
    re.compile(rf"\b{p}\b", re.IGNORECASE)
    for p in ["could of", "should of", "would of", "must of", "might of"]
]

_NOT_JUST_BUT_RE = re.compile(r"\bnot\s+(?:just|only)\b.{0,60}?\bbut\b(?:\s+also)?", re.IGNORECASE)

# Approximate "A, B, and C" / "A, B and C" list finder. Group 3 is the
# comma immediately before and/or -- its presence is the Oxford comma.
_LIST_RE = re.compile(
    r"([A-Za-z][\w\s]{0,30}),\s*([A-Za-z][\w\s]{0,30}?)(,)?\s+(and|or)\s+([A-Za-z][\w\s]{0,30})",
    re.IGNORECASE,
)

_CLAUSE_DEPS = {"ROOT", "conj", "ccomp", "advcl", "relcl", "acl", "xcomp", "csubj", "parataxis"}
_SUBORDINATE_DEPS = {"advcl", "ccomp", "relcl", "acl", "xcomp", "csubj", "mark"}
_FUNCTION_POS = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "AUX", "PART"}


@dataclass(frozen=True)
class DocumentStyloFeatures:
    doc_id: str
    n_sentences: int
    n_words: int
    sent_len_mean: float
    sent_len_std: float
    sent_len_burstiness: float
    mattr: float
    hapax_rate: float
    function_word_rate: float
    pos_trigram_entropy: float
    clause_count_mean: float
    subordination_ratio_mean: float
    em_dash_rate: float
    semicolon_rate: float
    oxford_comma_rate: float
    n_lists_detected: int
    construction_template_rate: float
    discourse_marker_rate: float
    spelling_error_rate: float
    homophone_error_rate: float
    proper_noun_rate: float
    numeral_rate: float
    named_entity_rate: float


def _mattr(words: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio."""
    if len(words) < 2:
        return 0.0
    if len(words) <= window:
        return len(set(words)) / len(words)
    ratios = [
        len(set(words[i : i + window])) / window for i in range(len(words) - window + 1)
    ]
    return sum(ratios) / len(ratios)


def _hapax_rate(words: list[str]) -> float:
    if not words:
        return 0.0
    counts = Counter(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    return hapax / len(words)


def _pos_trigram_entropy(pos_seq: list[str]) -> float:
    if len(pos_seq) < 3:
        return 0.0
    trigrams = list(zip(pos_seq, pos_seq[1:], pos_seq[2:], strict=False))
    counts = Counter(trigrams)
    total = len(trigrams)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _list_and_oxford_stats(text: str) -> tuple[int, int]:
    matches = list(_LIST_RE.finditer(text))
    n_oxford = sum(1 for m in matches if m.group(3))
    return len(matches), n_oxford


def _participial_opener_count(sentences_tokens: list[list]) -> int:  # noqa: ANN401
    count = 0
    for tokens in sentences_tokens:
        if not tokens or tokens[0].tag_ not in ("VBG", "VBN"):
            continue
        if any(t.text == "," for t in tokens[1:8]):
            count += 1
    return count


def compute_document_features(doc_id: str, text: str) -> DocumentStyloFeatures:
    nlp = get_nlp()
    paragraphs = split_paragraphs(text)

    all_words: list[str] = []
    checkable_words: list[str] = []
    all_pos: list[str] = []
    sent_lengths: list[int] = []
    sentences_tokens: list[list] = []  # noqa: ANN401
    clause_counts: list[int] = []
    subordination_ratios: list[float] = []
    n_propn = 0
    n_num = 0
    n_ents = 0

    for para in paragraphs:
        spacy_doc = nlp(para)
        n_ents += len(spacy_doc.ents)
        for sent in spacy_doc.sents:
            sent_tokens = [t for t in sent if not t.is_space]
            if not sent_tokens:
                continue
            sentences_tokens.append(sent_tokens)
            sent_lengths.append(sum(1 for t in sent_tokens if t.is_alpha))

            clause_anchors = [
                t for t in sent_tokens if t.pos_ in ("VERB", "AUX") and t.dep_ in _CLAUSE_DEPS
            ]
            subordinate_anchors = [t for t in sent_tokens if t.dep_ in _SUBORDINATE_DEPS]
            n_clauses = max(len(clause_anchors), 1)
            clause_counts.append(n_clauses)
            subordination_ratios.append(len(subordinate_anchors) / n_clauses)

            for t in sent_tokens:
                if t.is_alpha:
                    all_words.append(t.text.lower())
                    all_pos.append(t.pos_)
                    if t.pos_ != "PROPN" and len(t.text) > 1:
                        checkable_words.append(t.text.lower())
                if t.pos_ == "PROPN":
                    n_propn += 1
                if t.like_num:
                    n_num += 1

    n_words = len(all_words)
    n_sentences = len(sent_lengths)
    per100 = 100.0 / n_words if n_words else 0.0

    sent_len_mean = sum(sent_lengths) / n_sentences if n_sentences else 0.0
    sent_len_std = (
        math.sqrt(sum((x - sent_len_mean) ** 2 for x in sent_lengths) / n_sentences)
        if n_sentences
        else 0.0
    )
    burstiness = sent_len_std / sent_len_mean if sent_len_mean else 0.0

    n_function = sum(1 for p in all_pos if p in _FUNCTION_POS)
    function_word_rate = n_function / n_words if n_words else 0.0

    n_lists, n_oxford = _list_and_oxford_stats(text)
    oxford_comma_rate = (n_oxford / n_lists) if n_lists else -1.0  # -1 = not applicable

    n_participial = _participial_opener_count(sentences_tokens)
    n_not_just_but = len(_NOT_JUST_BUT_RE.findall(text))
    construction_hits = n_not_just_but + n_lists + n_participial

    n_homophone_errors = sum(len(p.findall(text)) for p in _HOMOPHONE_ERROR_PATTERNS)

    unknown = _SPELL.unknown(checkable_words) if checkable_words else set()
    spelling_error_rate = (len(unknown) / len(checkable_words)) if checkable_words else 0.0

    n_discourse = sum(1 for w in all_words if w in DISCOURSE_MARKERS)

    em_dash_count = text.count("—") + text.count("--")
    semicolon_count = text.count(";")

    return DocumentStyloFeatures(
        doc_id=doc_id,
        n_sentences=n_sentences,
        n_words=n_words,
        sent_len_mean=sent_len_mean,
        sent_len_std=sent_len_std,
        sent_len_burstiness=burstiness,
        mattr=_mattr(all_words),
        hapax_rate=_hapax_rate(all_words),
        function_word_rate=function_word_rate,
        pos_trigram_entropy=_pos_trigram_entropy(all_pos),
        clause_count_mean=sum(clause_counts) / len(clause_counts) if clause_counts else 0.0,
        subordination_ratio_mean=(
            sum(subordination_ratios) / len(subordination_ratios)
            if subordination_ratios
            else 0.0
        ),
        em_dash_rate=em_dash_count * per100,
        semicolon_rate=semicolon_count * per100,
        oxford_comma_rate=oxford_comma_rate,
        n_lists_detected=n_lists,
        construction_template_rate=construction_hits * per100,
        discourse_marker_rate=n_discourse * per100,
        spelling_error_rate=spelling_error_rate,
        homophone_error_rate=n_homophone_errors * per100,
        proper_noun_rate=n_propn * per100,
        numeral_rate=n_num * per100,
        named_entity_rate=n_ents * per100,
    )

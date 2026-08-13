"""Sentence and paragraph segmentation (Module 2). Sentence is the
addressable span for every downstream feature (ADR 0002); this module
turns raw essay text into that per-sentence structure.

Distinct from `src/detector/segmentation/`, which is Module 5's
smoothing + change-point detection over already-scored sentences.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import spacy
from spacy.language import Language

_NLP: Language | None = None


def get_nlp() -> Language:
    """Shared pipeline. Keeps parser (clauses/subordination), tagger
    (POS/function words), and NER (concreteness) -- all needed by the
    Module 2 stylometric features. Only lemmatizer is dropped (unused).
    """
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", exclude=["lemmatizer"])
    return _NLP


@dataclass(frozen=True)
class Sentence:
    doc_id: str
    paragraph_index: int
    sentence_index: int
    text: str
    word_count: int


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def segment_document(doc_id: str, text: str) -> Iterator[Sentence]:
    """Yield every sentence in a document, tagged with its paragraph and
    sentence index so downstream features can address either span.
    """
    nlp = get_nlp()
    paragraphs = split_paragraphs(text)
    for para_idx, paragraph in enumerate(paragraphs):
        spacy_doc = nlp(paragraph)
        for sent_idx, sent in enumerate(spacy_doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            yield Sentence(
                doc_id=doc_id,
                paragraph_index=para_idx,
                sentence_index=sent_idx,
                text=sent_text,
                word_count=len(sent_text.split()),
            )


def segment_documents_batch(
    doc_id_and_text: Iterable[tuple[str, str]], batch_size: int = 128, n_process: int = 1
) -> Iterator[Sentence]:
    """Same output as calling segment_document per document, but processes
    every paragraph across every document in one nlp.pipe() batch -- far
    faster than one nlp() call per paragraph (per-call pipeline overhead
    dominates at this scale: ~5,450 documents was 16+ minutes unbatched).
    """
    nlp = get_nlp()

    # Flatten to (doc_id, paragraph_index, paragraph_text) so nlp.pipe can
    # batch across paragraph AND document boundaries at once.
    flat: list[tuple[str, int]] = []
    texts: list[str] = []
    for doc_id, text in doc_id_and_text:
        for para_idx, paragraph in enumerate(split_paragraphs(text)):
            flat.append((doc_id, para_idx))
            texts.append(paragraph)

    for (doc_id, para_idx), spacy_doc in zip(
        flat, nlp.pipe(texts, batch_size=batch_size, n_process=n_process), strict=True
    ):
        for sent_idx, sent in enumerate(spacy_doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue
            yield Sentence(
                doc_id=doc_id,
                paragraph_index=para_idx,
                sentence_index=sent_idx,
                text=sent_text,
                word_count=len(sent_text.split()),
            )

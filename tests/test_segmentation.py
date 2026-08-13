from detector.ingest.segmentation import (
    segment_document,
    segment_documents_batch,
    split_paragraphs,
)


def test_split_paragraphs_on_blank_lines() -> None:
    text = "Para one.\nStill para one.\n\nPara two."
    assert split_paragraphs(text) == ["Para one.\nStill para one.", "Para two."]


def test_split_paragraphs_drops_empty_chunks() -> None:
    assert split_paragraphs("\n\nOnly content.\n\n\n") == ["Only content."]


def test_segment_document_indexes_paragraphs_and_sentences() -> None:
    text = "First sentence. Second sentence.\n\nThird sentence here."
    sentences = list(segment_document("doc1", text))

    assert [s.paragraph_index for s in sentences] == [0, 0, 1]
    assert [s.sentence_index for s in sentences] == [0, 1, 0]
    assert all(s.doc_id == "doc1" for s in sentences)
    assert sentences[0].text == "First sentence."
    assert sentences[2].text == "Third sentence here."


def test_segment_document_word_count() -> None:
    sentences = list(segment_document("doc1", "One two three four."))
    assert sentences[0].word_count == 4


def test_segment_document_empty_text_yields_nothing() -> None:
    assert list(segment_document("doc1", "")) == []


def test_segment_documents_batch_matches_per_document_output() -> None:
    docs = [
        ("doc1", "First sentence. Second sentence.\n\nThird sentence here."),
        ("doc2", "Only one sentence."),
    ]
    batched = list(segment_documents_batch(docs))
    individually = [s for doc_id, text in docs for s in segment_document(doc_id, text)]

    assert batched == individually


def test_segment_documents_batch_handles_empty_document() -> None:
    docs = [("doc1", "Has content."), ("doc2", "")]
    batched = list(segment_documents_batch(docs))
    assert [s.doc_id for s in batched] == ["doc1"]

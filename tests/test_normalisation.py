from detector.ingest.normalisation import normalize_text, text_hash


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("Hello   world\n\nfoo") == "hello world foo"


def test_normalize_text_strips_and_lowercases() -> None:
    assert normalize_text("  Mixed CASE Text  ") == "mixed case text"


def test_normalize_text_idempotent() -> None:
    once = normalize_text("A  B\tC")
    assert normalize_text(once) == once


def test_text_hash_stable_across_whitespace_variants() -> None:
    a = "Hello   world.\nSecond sentence."
    b = "hello world. second sentence."
    assert text_hash(a) == text_hash(b)


def test_text_hash_differs_for_different_text() -> None:
    assert text_hash("hello world") != text_hash("hello there")

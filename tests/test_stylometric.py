from detector.features.stylometric import (
    _hapax_rate,
    _list_and_oxford_stats,
    _mattr,
    _pos_trigram_entropy,
    compute_document_features,
)


def test_mattr_full_ttr_when_shorter_than_window() -> None:
    words = ["a", "b", "a", "c"]
    assert _mattr(words, window=50) == len(set(words)) / len(words)


def test_mattr_repeated_word_is_lower_than_all_unique() -> None:
    repeated = ["the"] * 20
    unique = [f"word{i}" for i in range(20)]
    assert _mattr(repeated, window=5) < _mattr(unique, window=5)


def test_hapax_rate_all_unique_is_one() -> None:
    assert _hapax_rate(["a", "b", "c"]) == 1.0


def test_hapax_rate_all_repeated_is_zero() -> None:
    assert _hapax_rate(["a", "a", "a"]) == 0.0


def test_hapax_rate_empty_is_zero() -> None:
    assert _hapax_rate([]) == 0.0


def test_pos_trigram_entropy_uniform_sequence_is_zero() -> None:
    # every trigram identical -> zero entropy
    assert _pos_trigram_entropy(["NOUN", "VERB", "NOUN"] * 5) >= 0.0
    assert _pos_trigram_entropy(["NOUN"] * 10) == 0.0


def test_pos_trigram_entropy_short_sequence_is_zero() -> None:
    assert _pos_trigram_entropy(["NOUN", "VERB"]) == 0.0


def test_list_and_oxford_stats_detects_oxford_comma() -> None:
    text = "We bought apples, oranges, and pears."
    n_lists, n_oxford = _list_and_oxford_stats(text)
    assert n_lists == 1
    assert n_oxford == 1


def test_list_and_oxford_stats_detects_missing_oxford_comma() -> None:
    text = "We bought apples, oranges and pears."
    n_lists, n_oxford = _list_and_oxford_stats(text)
    assert n_lists == 1
    assert n_oxford == 0


def test_compute_document_features_discourse_marker_detected() -> None:
    text = "The plan failed. Moreover, nobody noticed until it was too late."
    features = compute_document_features("doc1", text)
    assert features.discourse_marker_rate > 0


def test_compute_document_features_homophone_error_detected() -> None:
    text = "I could of finished earlier if I had started sooner."
    features = compute_document_features("doc1", text)
    assert features.homophone_error_rate > 0


def test_compute_document_features_em_dash_and_semicolon_rates() -> None:
    text = "This works well — better than expected; we should continue."
    features = compute_document_features("doc1", text)
    assert features.em_dash_rate > 0
    assert features.semicolon_rate > 0


def test_compute_document_features_empty_text_does_not_crash() -> None:
    features = compute_document_features("doc1", "")
    assert features.n_words == 0
    assert features.n_sentences == 0
    assert features.oxford_comma_rate == -1.0


def test_compute_document_features_oxford_comma_sentinel_when_no_lists() -> None:
    text = "This is a plain sentence with no lists at all."
    features = compute_document_features("doc1", text)
    assert features.n_lists_detected == 0
    assert features.oxford_comma_rate == -1.0

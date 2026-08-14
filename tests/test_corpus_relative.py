import pandas as pd

from detector.features.corpus_relative import (
    build_baseline_stats,
    intra_document_consistency,
    length_band,
    zscore_features,
)


def test_length_band_short() -> None:
    assert length_band(100) == "short"
    assert length_band(249) == "short"


def test_length_band_medium() -> None:
    assert length_band(250) == "medium"
    assert length_band(499) == "medium"


def test_length_band_long() -> None:
    assert length_band(500) == "long"
    assert length_band(10000) == "long"


def test_build_baseline_stats_groups_by_band() -> None:
    baseline = pd.DataFrame(
        {
            "word_count": [100, 120, 110, 600, 620, 610],
            "feat": [1.0, 1.2, 0.9, 5.0, 5.2, 4.9],
        }
    )
    stats = build_baseline_stats(baseline, ["feat"])
    assert {"short", "long"}.issubset(set(stats.index))
    assert stats.loc["short", "feat__mean"] < stats.loc["long", "feat__mean"]


def test_zscore_features_uses_matching_band() -> None:
    baseline = pd.DataFrame(
        {
            "word_count": [100, 120, 110, 600, 620, 610],
            "feat": [1.0, 1.2, 0.9, 5.0, 5.2, 4.9],
        }
    )
    stats = build_baseline_stats(baseline, ["feat"])

    new = pd.DataFrame({"word_count": [105], "feat": [1.033]})
    z = zscore_features(new, ["feat"], stats)
    assert abs(z["feat_z"].iloc[0]) < 0.5  # close to the short-band mean


def test_zscore_features_falls_back_when_band_missing() -> None:
    baseline = pd.DataFrame({"word_count": [100, 120, 110], "feat": [1.0, 1.2, 0.9]})
    stats = build_baseline_stats(baseline, ["feat"])

    # no "long" band in baseline_stats -- must fall back to pooled stats, not crash
    new = pd.DataFrame({"word_count": [700], "feat": [5.0]})
    z = zscore_features(new, ["feat"], stats)
    assert not z["feat_z"].isna().any()


def test_intra_document_consistency_zero_for_uniform_document() -> None:
    df = pd.DataFrame(
        {
            "doc_id": ["a", "a", "a"],
            "feat": [1.0, 1.0, 1.0],
        }
    )
    result = intra_document_consistency(df, ["feat"])
    assert (result["intra_doc_consistency"] == 0.0).all()


def test_intra_document_consistency_nonzero_for_varied_document() -> None:
    df = pd.DataFrame(
        {
            "doc_id": ["a", "a", "a", "b", "b", "b"],
            "feat": [1.0, 1.0, 1.0, 1.0, 5.0, 9.0],
        }
    )
    result = intra_document_consistency(df, ["feat"])
    doc_a = result[result["doc_id"] == "a"]["intra_doc_consistency"]
    doc_b = result[result["doc_id"] == "b"]["intra_doc_consistency"]
    assert doc_a.mean() < doc_b.mean()

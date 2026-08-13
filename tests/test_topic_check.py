import pandas as pd

from detector.ingest.topic_check import topic_distribution_comparison


def test_identical_distributions_have_zero_gap() -> None:
    manifest = pd.DataFrame(
        {
            "author_type": ["human", "human", "machine", "machine"],
            "topic": ["A", "B", "A", "B"],
        }
    )
    comparison = topic_distribution_comparison(manifest)
    assert (comparison["gap_pct"] < 1e-9).all()


def test_flags_large_gap() -> None:
    manifest = pd.DataFrame(
        {
            "author_type": ["human"] * 90 + ["human"] * 10 + ["machine"] * 100,
            "topic": ["A"] * 90 + ["B"] * 10 + ["A"] * 100,
        }
    )
    comparison = topic_distribution_comparison(manifest)
    # topic B: 10% human, 0% machine -> 10 pp gap
    assert comparison.loc["B", "gap_pct"] > 5.0


def test_missing_topic_on_one_side_treated_as_zero() -> None:
    manifest = pd.DataFrame(
        {
            "author_type": ["human", "machine"],
            "topic": ["only_human", "only_machine"],
        }
    )
    comparison = topic_distribution_comparison(manifest)
    assert comparison.loc["only_human", "machine_pct"] == 0.0
    assert comparison.loc["only_machine", "human_pct"] == 0.0

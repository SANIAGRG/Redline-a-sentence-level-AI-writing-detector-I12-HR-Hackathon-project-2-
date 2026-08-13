import pandas as pd

from detector.ingest.sampling import HELD_OUT_FAMILY, sample_human_pools, sample_machine_pools


def _synthetic_joined(n_yes: int, n_no: int, n_unknown: int) -> pd.DataFrame:
    statuses = ["Yes"] * n_yes + ["No"] * n_no + [None] * n_unknown
    rows = [
        {"text_hash": f"h{i}", "ell_status": status, "word_count": 300}
        for i, status in enumerate(statuses)
    ]
    return pd.DataFrame(rows)


def _synthetic_daigt(
    family_sizes: dict[str, int], n_human: int = 50, topics: tuple[str, ...] = ("A", "B")
) -> pd.DataFrame:
    rows = []
    i = 0
    for source, n in family_sizes.items():
        for j in range(n):
            rows.append(
                {
                    "text": f"essay {i}",
                    "label": 1,
                    "source": source,
                    "prompt_name": topics[j % len(topics)],
                }
            )
            i += 1
    for _ in range(n_human):
        rows.append(
            {
                "text": f"human {i}",
                "label": 0,
                "source": "persuade_corpus",
                "prompt_name": topics[0],
            }
        )
        i += 1
    return pd.DataFrame(rows)


_EVEN_TOPIC_SHARE = pd.Series({"A": 0.5, "B": 0.5})


def test_sample_human_pools_sizes_and_balance() -> None:
    joined = _synthetic_joined(n_yes=200, n_no=800, n_unknown=50)

    baseline, training = sample_human_pools(joined, baseline_n=500, training_n=100, seed=1)

    assert len(baseline) == 500
    assert len(training) == 100
    ell = training["ell_status"]
    assert (ell == "Yes").sum() == 50
    assert (ell == "No").sum() == 50


def test_sample_human_pools_disjoint() -> None:
    joined = _synthetic_joined(n_yes=200, n_no=800, n_unknown=50)

    baseline, training = sample_human_pools(joined, baseline_n=500, training_n=100, seed=1)

    assert set(baseline["text_hash"]).isdisjoint(set(training["text_hash"]))


def test_sample_human_pools_deterministic() -> None:
    joined = _synthetic_joined(n_yes=200, n_no=800, n_unknown=50)

    b1, t1 = sample_human_pools(joined, baseline_n=100, training_n=40, seed=7)
    b2, t2 = sample_human_pools(joined, baseline_n=100, training_n=40, seed=7)

    assert set(b1["text_hash"]) == set(b2["text_hash"])
    assert set(t1["text_hash"]) == set(t2["text_hash"])


def test_sample_machine_pools_excludes_held_out_family_from_training() -> None:
    daigt = _synthetic_daigt({HELD_OUT_FAMILY: 300, "family_a": 200, "family_b": 200})

    training, heldout = sample_machine_pools(
        daigt, _EVEN_TOPIC_SHARE, training_n=100, heldout_n=50, seed=1
    )

    assert HELD_OUT_FAMILY not in set(training["source"])
    assert set(heldout["source"]) == {HELD_OUT_FAMILY}
    assert len(heldout) == 50


def test_sample_machine_pools_represents_multiple_families() -> None:
    daigt = _synthetic_daigt({HELD_OUT_FAMILY: 300, "family_a": 200, "family_b": 200})

    training, _ = sample_machine_pools(
        daigt, _EVEN_TOPIC_SHARE, training_n=100, heldout_n=50, seed=1
    )

    assert set(training["source"]) == {"family_a", "family_b"}
    assert len(training) <= 100


def test_sample_machine_pools_ignores_human_rows() -> None:
    daigt = _synthetic_daigt({HELD_OUT_FAMILY: 300, "family_a": 200}, n_human=100)

    training, heldout = sample_machine_pools(
        daigt, _EVEN_TOPIC_SHARE, training_n=50, heldout_n=20, seed=1
    )

    assert "persuade_corpus" not in set(training["source"]) | set(heldout["source"])


def test_sample_machine_pools_matches_topic_share_despite_family_skew() -> None:
    # family_a is 100% topic A, family_b is 100% topic B -- a naive
    # family-proportional sample would badly skew topic coverage, but
    # topic stratification should still hit the requested 50/50 split.
    rows = []
    for i in range(500):
        rows.append({"text": f"e{i}", "label": 1, "source": "family_a", "prompt_name": "A"})
    for i in range(500, 1000):
        rows.append({"text": f"e{i}", "label": 1, "source": "family_b", "prompt_name": "B"})
    daigt = pd.DataFrame(rows)

    training, _ = sample_machine_pools(
        daigt, _EVEN_TOPIC_SHARE, training_n=200, heldout_n=0, seed=1
    )

    topic_counts = training["prompt_name"].value_counts(normalize=True)
    assert abs(topic_counts["A"] - 0.5) < 0.05
    assert abs(topic_counts["B"] - 0.5) < 0.05


def test_sample_machine_pools_skewed_target_share_is_respected() -> None:
    daigt = _synthetic_daigt({"family_a": 1000}, n_human=0)
    skewed_share = pd.Series({"A": 0.9, "B": 0.1})

    training, _ = sample_machine_pools(daigt, skewed_share, training_n=200, heldout_n=0, seed=1)

    topic_counts = training["prompt_name"].value_counts(normalize=True)
    assert topic_counts["A"] > topic_counts["B"]

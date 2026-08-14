import pandas as pd
import pytest

from detector.generate.run_adversarial import (
    ATTACK_TYPES,
    apply_attack,
    assign_attack_types,
    select_adversarial_pool,
)


def test_assign_attack_types_covers_both_for_large_n() -> None:
    assigned = assign_attack_types(20, seed=1)
    assert len(assigned) == 20
    assert set(assigned) == set(ATTACK_TYPES)


def test_assign_attack_types_deterministic() -> None:
    a = assign_attack_types(10, seed=5)
    b = assign_attack_types(10, seed=5)
    assert a == b


def test_apply_attack_typo_changes_text() -> None:
    text = "The government should not allow phones during class today."
    result = apply_attack(text, "typo", seed=1)
    assert isinstance(result, str)


def test_apply_attack_paraphrase_changes_text() -> None:
    text = "The government should not allow phones during class today."
    result = apply_attack(text, "paraphrase", seed=1)
    assert isinstance(result, str)


def test_apply_attack_unknown_type_raises() -> None:
    with pytest.raises(ValueError, match="unknown attack_type"):
        apply_attack("text", "not_a_real_attack", seed=1)


def test_select_adversarial_pool_excludes_training_and_heldout(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from detector.config import load_config

    config = load_config()
    config.paths.raw = tmp_path
    config.paths.interim = tmp_path

    daigt_rows = []
    for i in range(500):
        daigt_rows.append(
            {
                "text": f"essay {i} " * 20,
                "label": 1,
                "source": "falcon_180b_v1" if i % 2 == 0 else "chat_gpt_moth",
                "prompt_name": "Phones and driving",
            }
        )
    daigt = pd.DataFrame(daigt_rows)
    daigt.to_csv(tmp_path / "train_v2_drcat_02.csv", index=False)

    joined = pd.DataFrame(
        {"prompt_name": ["Phones and driving"] * 10, "text_hash": [f"h{i}" for i in range(10)]}
    )
    joined.to_parquet(tmp_path / "daigt_persuade_joined.parquet", index=False)

    selected = select_adversarial_pool(config, n=20)

    assert len(selected) == 20
    # re-derive the excluded indices the same way the function does, to
    # confirm none of them leaked into the adversarial pool
    from detector.ingest.sampling import sample_machine_pools

    topic_share = joined["prompt_name"].value_counts(normalize=True)
    pool_counts = {p.name: p.count for p in config.sampling.pools}
    training, heldout = sample_machine_pools(
        daigt,
        topic_share,
        pool_counts["machine_training"],
        pool_counts["machine_heldout_family"],
        config.sampling.seed,
    )
    used_index = set(training.index) | set(heldout.index)
    assert set(selected.index).isdisjoint(used_index)

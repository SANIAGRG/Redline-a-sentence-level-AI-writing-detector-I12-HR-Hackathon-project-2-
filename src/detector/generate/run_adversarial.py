"""Resumable adversarial-set runner (Module 3C). Eval-only -- these rows
are never trained on (spec is explicit about this).

Samples 100 existing DAIGT machine essays, disjoint from the
machine_training and machine_heldout_family pools (Module 2) so no essay
is evaluated twice under different conditions. Applies one of two
attacks per essay (typo injection or WordNet synonym-substitution
paraphrase, ~50/50) via `detector.generate.adversarial`. No Ollama --
H2 reserves it for the polish corpus and modern-generator slice only.

Checkpointing (H4): same append-after-every-essay pattern as the other
Module 3 runners. `make generate` runs this after run_polish/run_modern.
Standalone: `python -m detector.generate.run_adversarial`.
"""

from __future__ import annotations

import logging
import random

import pandas as pd

from detector.config import RedlineConfig, load_config
from detector.generate.adversarial import inject_typos, paraphrase_synonyms
from detector.generate.checkpoint import append_checkpoint, load_checkpoint
from detector.ingest.loaders import load_daigt
from detector.ingest.sampling import sample_machine_pools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ATTACK_TYPES = ("typo", "paraphrase")
TYPO_RATE = 0.12
PARAPHRASE_RATE = 0.3
CHECKPOINT_COLUMNS = [
    "essay_id",
    "source_family",
    "topic",
    "attack_type",
    "original_text",
    "attacked_text",
    "word_count",
]


def select_adversarial_pool(config: RedlineConfig, n: int) -> pd.DataFrame:
    """Sample n machine essays disjoint from machine_training and
    machine_heldout_family (recomputed with the same seed, not read back
    from a manifest file, so this has no dependency on `make sample`
    having been run first).
    """
    daigt = load_daigt(config.paths.raw)
    joined = pd.read_parquet(config.paths.interim / "daigt_persuade_joined.parquet")
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

    machine = daigt[(daigt["label"] == 1) & (~daigt["source"].isin({"persuade_corpus"}))]
    eligible = machine[~machine.index.isin(used_index)]
    return eligible.sample(n=min(n, len(eligible)), random_state=config.sampling.seed)


def assign_attack_types(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    base = list(ATTACK_TYPES) * (n // len(ATTACK_TYPES) + 1)
    rng.shuffle(base)
    return base[:n]


def apply_attack(text: str, attack_type: str, seed: int) -> str:
    if attack_type == "typo":
        return inject_typos(text, rate=TYPO_RATE, seed=seed)
    if attack_type == "paraphrase":
        return paraphrase_synonyms(text, rate=PARAPHRASE_RATE, seed=seed)
    raise ValueError(f"unknown attack_type: {attack_type}")


def run(config: RedlineConfig, target_n: int | None = None) -> None:
    pool_count = next((p.count for p in config.sampling.pools if p.name == "adversarial"), 100)
    n = target_n or pool_count

    out_dir = config.paths.generated / "adversarial"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.parquet"

    selected = select_adversarial_pool(config, n)
    attack_types = assign_attack_types(len(selected), seed=config.sampling.seed)
    selected = selected.copy()
    selected["essay_id"] = selected["source"].astype(str) + "::adv::" + selected.index.astype(str)
    selected["attack_type"] = attack_types

    done = set(load_checkpoint(checkpoint_path, CHECKPOINT_COLUMNS)["essay_id"])
    remaining = selected[~selected["essay_id"].isin(done)]
    logger.info(
        "Adversarial run: %d target, %d already done, %d remaining.",
        len(selected),
        len(done),
        len(remaining),
    )

    for i, (_, row) in enumerate(remaining.iterrows(), start=1):
        seed = hash(row["essay_id"]) % (2**31)
        attacked_text = apply_attack(row["text"], row["attack_type"], seed=seed)

        checkpoint_row = {
            "essay_id": row["essay_id"],
            "source_family": row["source"],
            "topic": row["prompt_name"],
            "attack_type": row["attack_type"],
            "original_text": row["text"],
            "attacked_text": attacked_text,
            "word_count": len(attacked_text.split()),
        }
        append_checkpoint(checkpoint_path, checkpoint_row, CHECKPOINT_COLUMNS)
        logger.info(
            "[%d/%d] attacked essay_id=%s attack_type=%s",
            i,
            len(remaining),
            row["essay_id"],
            row["attack_type"],
        )

    n_total = len(load_checkpoint(checkpoint_path, CHECKPOINT_COLUMNS))
    logger.info("Adversarial run complete. %d essays in checkpoint.", n_total)


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

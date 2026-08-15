"""Scores both the original and attacked text of every adversarial essay
(Module 3C, 100 essays) through the same stylometric + likelihood
pipeline used for training, so degradation (score(attacked) vs.
score(original)) can be measured. Checkpointed like the other Module 4
runners -- resumable.
"""

from __future__ import annotations

import json
import logging

import pandas as pd

from detector.config import RedlineConfig, load_config
from detector.features.likelihood import compute_document_likelihood
from detector.features.stylometric import compute_document_features
from detector.generate.checkpoint import append_checkpoint, load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LIKELIHOOD_COLS = [
    "mean_logprob",
    "mean_logrank",
    "lrr",
    "mean_entropy",
    "mean_curvature",
    "logprob_variance",
    "cross_ppl_ratio",
]

CHECKPOINT_COLUMNS = ["doc_id", "essay_id", "version", "attack_type"] + LIKELIHOOD_COLS + ["stylo_json"]


def run(config: RedlineConfig) -> None:
    checkpoint = pd.read_parquet(config.paths.generated / "adversarial" / "checkpoint.parquet")

    out_dir = config.paths.features
    checkpoint_path = out_dir / "adversarial_features_checkpoint.parquet"

    targets = []
    for _, row in checkpoint.iterrows():
        eid, atk = row["essay_id"], row["attack_type"]
        targets.append((f"{eid}::orig", eid, "original", atk, row["original_text"]))
        targets.append((f"{eid}::attk", eid, "attacked", atk, row["attacked_text"]))

    done = set(load_checkpoint(checkpoint_path, CHECKPOINT_COLUMNS)["doc_id"])
    remaining = [t for t in targets if t[0] not in done]
    logger.info(
        "Adversarial feature run: %d target, %d done, %d remaining.", len(targets), len(done), len(remaining)
    )

    for i, (doc_id, essay_id, version, attack_type, text) in enumerate(remaining, start=1):
        try:
            likelihood_results = compute_document_likelihood(doc_id, text, seed=config.sampling.seed)
            stylo = compute_document_features(doc_id, text)
        except Exception:
            logger.exception("Failed on doc_id=%s -- will retry.", doc_id)
            continue

        if not likelihood_results:
            continue
        n = len(likelihood_results)
        means = {c: sum(getattr(r, c) for r in likelihood_results) / n for c in LIKELIHOOD_COLS}

        row = {
            "doc_id": doc_id,
            "essay_id": essay_id,
            "version": version,
            "attack_type": attack_type,
            **means,
            "stylo_json": json.dumps(stylo.__dict__),
        }
        append_checkpoint(checkpoint_path, row, CHECKPOINT_COLUMNS)
        logger.info("[%d/%d] scored doc_id=%s", i, len(remaining), doc_id)

    logger.info("Adversarial feature run complete.")


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

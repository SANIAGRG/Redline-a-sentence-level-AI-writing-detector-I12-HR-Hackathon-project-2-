"""Module 4 driver: run likelihood scoring (7A) across a deadline-scoped
slice of every pool, then build the corpus-relative z-scored feature
frame (7C, minus the embedding-centroid signal -- cut, see ADR 0008).

Pool sizes here are far below the original spec targets -- a
measured-and-documented compute-budget cut (ADR 0008), not an oversight:
baseline 15 (was 4,000 in spec, 500 after the first cut), human_training
25 (was 600), machine_training 25 (was 700), machine_heldout 20 (was
150), polished 60 (was 350, floor per explicit instruction), modern_gen
45 (all of it, capped from 150 in Module 3).

Each essay is scored on 3 sampled sentences (1 early/mid/late), not all
of them -- see docs/detector/features/likelihood.py docstring.

Checkpointed the same way as the Module 3 generation runners --
resumable, appends after every document.
"""

from __future__ import annotations

import logging

import pandas as pd

from detector.config import RedlineConfig, load_config
from detector.features.likelihood import compute_document_likelihood
from detector.generate.checkpoint import append_checkpoint, load_checkpoint
from detector.ingest.loaders import load_daigt
from detector.ingest.sampling import sample_human_pools, sample_machine_pools

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POOL_SIZES = {
    "baseline": 15,
    "human_training": 25,
    "machine_training": 25,
    "machine_heldout": 20,
    "polished": 60,
    "modern_gen": 45,
}

CHECKPOINT_COLUMNS = [
    "doc_id",
    "pool",
    "paragraph_index",
    "sentence_index",
    "text",
    "mean_logprob",
    "mean_lograsnk",
    "lrr",
    "mean_entropy",
    "mean_curvature",
    "logprob_variance",
    "cross_ppl_ratio",
]


def build_documents(config: RedlineConfig) -> list[tuple[str, str, str, int]]:
    """Returns (doc_id, text, pool, word_count) for every document to
    score, across all 6 pools, sized per POOL_SIZES.
    """
    docs: list[tuple[str, str, str, int]] = []
    seed = config.sampling.seed

    joined = pd.read_parquet(config.paths.interim / "daigt_persuade_joined.parquet")
    baseline, human_training = sample_human_pools(
        joined, POOL_SIZES["baseline"], POOL_SIZES["human_training"], seed
    )
    for _, row in baseline.iterrows():
        docs.append((row["text_hash"], row["text"], "baseline", int(row["word_count"])))
    for _, row in human_training.iterrows():
        docs.append((row["text_hash"], row["text"], "human_training", int(row["word_count"])))

    daigt = load_daigt(config.paths.raw)
    topic_share = joined["prompt_name"].value_counts(normalize=True)
    machine_training, machine_heldout = sample_machine_pools(
        daigt, topic_share, POOL_SIZES["machine_training"], POOL_SIZES["machine_heldout"], seed
    )
    for idx, row in machine_training.iterrows():
        doc_id = f"{row['source']}::{idx}"
        docs.append((doc_id, row["text"], "machine_training", len(str(row["text"]).split())))
    for idx, row in machine_heldout.iterrows():
        doc_id = f"{row['source']}::{idx}"
        docs.append((doc_id, row["text"], "machine_heldout", len(str(row["text"]).split())))

    polish_checkpoint = pd.read_parquet(config.paths.generated / "polish" / "checkpoint.parquet")
    import json

    polish_sample = polish_checkpoint.sample(n=POOL_SIZES["polished"], random_state=seed)
    for _, row in polish_sample.iterrows():
        paras = json.loads(row["paragraphs_json"])
        full_text = "\n\n".join(p["revised_text"] for p in paras)
        wc = sum(len(p["revised_text"].split()) for p in paras)
        docs.append((row["essay_id"], full_text, "polished", wc))

    modern_checkpoint = pd.read_parquet(config.paths.generated / "modern" / "checkpoint.parquet")
    for _, row in modern_checkpoint.iterrows():
        docs.append((row["essay_id"], row["text"], "modern_gen", int(row["word_count"])))

    return docs


def run(config: RedlineConfig) -> None:
    out_dir = config.paths.features
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "likelihood_checkpoint.parquet"

    documents = build_documents(config)
    done_docs = set(load_checkpoint(checkpoint_path, CHECKPOINT_COLUMNS)["doc_id"])
    remaining = [d for d in documents if d[0] not in done_docs]
    logger.info(
        "Likelihood run: %d target docs, %d already done, %d remaining.",
        len(documents),
        len(documents) - len(remaining),
        len(remaining),
    )

    for i, (doc_id, text, pool, _word_count) in enumerate(remaining, start=1):
        try:
            results = compute_document_likelihood(doc_id, text, seed=config.sampling.seed)
        except Exception:
            logger.exception("Failed on doc_id=%s -- will retry next run.", doc_id)
            continue

        for r in results:
            row = {
                "doc_id": r.doc_id,
                "pool": pool,
                "paragraph_index": r.paragraph_index,
                "sentence_index": r.sentence_index,
                "text": r.text,
                "mean_logprob": r.mean_logprob,
                "mean_lograsnk": r.mean_lograsnk,
                "lrr": r.lrr,
                "mean_entropy": r.mean_entropy,
                "mean_curvature": r.mean_curvature,
                "logprob_variance": r.logprob_variance,
                "cross_ppl_ratio": r.cross_ppl_ratio,
            }
            append_checkpoint(checkpoint_path, row, CHECKPOINT_COLUMNS)

        logger.info(
            "[%d/%d] scored doc_id=%s pool=%s (%d sampled sentences)",
            i,
            len(remaining),
            doc_id[:16],
            pool,
            len(results),
        )

    logger.info("Likelihood run complete.")


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

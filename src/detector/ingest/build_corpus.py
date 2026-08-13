"""Module 2 driver: sample the four DAIGT/PERSUADE-sourced pools, segment
every document into sentences, write the manifest, and run the topic
check. `make sample` runs this.

Writes:
  dataset/interim/manifest.parquet
  dataset/interim/sentences.parquet
  docs/figures/topic_distribution.png
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

from detector.config import REPO_ROOT, RedlineConfig, load_config
from detector.ingest.manifest import build_manifest
from detector.ingest.sampling import SampledPools, build_sampled_pools
from detector.ingest.segmentation import segment_documents_batch
from detector.ingest.topic_check import (
    MAX_ACCEPTABLE_GAP_PCT,
    plot_topic_distribution,
    topic_distribution_comparison,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_sentences(manifest: pd.DataFrame, pools: SampledPools) -> pd.DataFrame:
    text_by_doc_id: dict[str, str] = {}
    for df, id_col, text_col in [
        (pools.human_baseline, "text_hash", "text"),
        (pools.human_training, "text_hash", "text"),
    ]:
        text_by_doc_id.update(zip(df[id_col], df[text_col], strict=True))
    for df in (pools.machine_training, pools.machine_heldout_family):
        doc_ids = df["source"].astype(str) + "::" + df.index.astype(str)
        text_by_doc_id.update(zip(doc_ids, df["text"], strict=True))

    doc_id_and_text = [(doc_id, text_by_doc_id[doc_id]) for doc_id in manifest["doc_id"]]
    rows = [s.__dict__ for s in segment_documents_batch(doc_id_and_text)]
    return pd.DataFrame(rows)


def run(config: RedlineConfig) -> int:
    interim_dir = config.paths.interim
    interim_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = REPO_ROOT / "docs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Sampling pools...")
    pools = build_sampled_pools(config)

    logger.info("Building manifest...")
    manifest = build_manifest(pools)
    manifest.to_parquet(interim_dir / "manifest.parquet", index=False)
    logger.info("Wrote %d manifest rows to %s", len(manifest), interim_dir / "manifest.parquet")

    logger.info("Segmenting documents into sentences...")
    sentences = build_sentences(manifest, pools)
    sentences.to_parquet(interim_dir / "sentences.parquet", index=False)
    logger.info(
        "Wrote %d sentences from %d documents to %s",
        len(sentences),
        manifest["doc_id"].nunique(),
        interim_dir / "sentences.parquet",
    )

    logger.info("Running topic distribution check...")
    comparison = topic_distribution_comparison(manifest)
    plot_topic_distribution(comparison, str(figures_dir / "topic_distribution.png"))
    max_gap = comparison["gap_pct"].max()
    logger.info("Max human/machine topic share gap: %.2f percentage points", max_gap)
    print(comparison.to_string())

    if max_gap > MAX_ACCEPTABLE_GAP_PCT:
        logger.warning(
            "Topic gap exceeds %.1f pp threshold -- classifier may learn topic, not authorship.",
            MAX_ACCEPTABLE_GAP_PCT,
        )
        return 1
    logger.info("Topic distributions are comparable (max gap %.2f pp).", max_gap)
    return 0


def main() -> None:
    sys.exit(run(load_config()))


if __name__ == "__main__":
    main()

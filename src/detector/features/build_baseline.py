"""Module 2 exit criterion: stylometric feature frame over the full
4,000-essay human baseline. `make stylometric` runs this.

Reads dataset/interim/manifest.parquet (written by `make sample`) and the
underlying joined text, recomputes document features for every
human_baseline row, and caches to dataset/features/stylometric_baseline.parquet.
"""

from __future__ import annotations

import logging

import pandas as pd

from detector.config import RedlineConfig, load_config
from detector.features.stylometric import compute_document_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(config: RedlineConfig) -> None:
    interim_dir = config.paths.interim
    features_dir = config.paths.features
    features_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(interim_dir / "manifest.parquet")
    baseline_manifest = manifest[manifest["pool"] == "human_baseline"]

    joined = pd.read_parquet(interim_dir / "daigt_persuade_joined.parquet")
    text_by_hash = dict(zip(joined["text_hash"], joined["text"], strict=True))

    rows = []
    for i, doc_id in enumerate(baseline_manifest["doc_id"]):
        if i % 500 == 0:
            logger.info("Stylometric features: %d/%d", i, len(baseline_manifest))
        text = text_by_hash[doc_id]
        rows.append(compute_document_features(doc_id, text).__dict__)

    frame = pd.DataFrame(rows)
    out_path = features_dir / "stylometric_baseline.parquet"
    frame.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(frame), out_path)


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

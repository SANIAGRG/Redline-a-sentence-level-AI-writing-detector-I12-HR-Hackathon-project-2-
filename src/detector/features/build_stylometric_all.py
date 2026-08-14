"""Stylometric features (7B) for the same deadline-scoped document set
used by build_likelihood.py, across all 6 pools -- Module 2 only
computed these for the (now-superseded) baseline pool. Cheap/fast, no
model, safe to run alongside the likelihood scoring pass.
"""

from __future__ import annotations

import logging

from detector.config import RedlineConfig, load_config
from detector.features.build_likelihood import build_documents
from detector.features.stylometric import compute_document_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(config: RedlineConfig) -> None:
    import pandas as pd

    documents = build_documents(config)
    rows = []
    for i, (doc_id, text, pool, _wc) in enumerate(documents):
        if i % 50 == 0:
            logger.info("Stylometric: %d/%d", i, len(documents))
        features = compute_document_features(doc_id, text)
        row = features.__dict__.copy()
        row["pool"] = pool
        rows.append(row)

    frame = pd.DataFrame(rows)
    out_path = config.paths.features / "stylometric_all.parquet"
    frame.to_parquet(out_path, index=False)
    logger.info("Wrote %d rows to %s", len(frame), out_path)


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

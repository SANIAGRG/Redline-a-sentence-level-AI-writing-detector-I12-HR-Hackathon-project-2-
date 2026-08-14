"""Module 5: L2-regularised logistic regression on z-scored features,
isotonic-calibrated. LightGBM comparison skipped under the deadline
(ADR 0008; spec's own first-cut list). Ships as the sole model --
coefficients read directly as evidence, which was always the primary
reason to prefer it, not just the head-to-head comparison.
"""

from __future__ import annotations

import logging

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from detector.config import RedlineConfig, load_config
from detector.model.build_training_frame import Z_COLS, build_zscored_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


TRAIN_POOLS = {"human_training", "machine_training"}


def run(config: RedlineConfig) -> None:
    frame = build_zscored_frame(config)
    # Only human_training/machine_training are ever trained on. baseline is
    # the z-score reference only; machine_heldout and modern_gen are eval-only
    # conditions (ADR 0007 -- modern_gen "held out entirely... never training
    # data") and must never leak into the train/test split, or the "held-out"
    # eval conditions evaluate.py reports would be scoring memorized essays.
    trainable = frame[frame["pool"].isin(TRAIN_POOLS)].copy()
    logger.info("Trainable rows: %d (%s)", len(trainable), trainable["pool"].value_counts().to_dict())

    X = trainable[Z_COLS].fillna(0.0).values
    y = trainable["label"].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, trainable.index, test_size=0.2, random_state=config.sampling.seed, stratify=y
    )

    base_model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=config.sampling.seed)

    n_per_class = min(np.bincount(y_train.astype(int)))
    cv = min(3, n_per_class) if n_per_class >= 2 else 2
    try:
        calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=cv)
        calibrated.fit(X_train, y_train)
    except Exception:
        logger.warning("Isotonic calibration failed (likely too little data) -- falling back to sigmoid.")
        calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv=cv)
        calibrated.fit(X_train, y_train)

    train_acc = calibrated.score(X_train, y_train)
    test_acc = calibrated.score(X_test, y_test)
    logger.info("Train accuracy: %.3f, held-out test accuracy: %.3f", train_acc, test_acc)

    out_dir = config.paths.features.parent / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": calibrated,
            "feature_cols": Z_COLS,
            "test_index": list(idx_test),
            "train_index": list(idx_train),
            "seed": config.sampling.seed,
        },
        out_dir / "logistic_regression.joblib",
    )
    logger.info("Saved model to %s", out_dir / "logistic_regression.joblib")


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

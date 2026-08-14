"""Module 5 evaluation: three test conditions (in-distribution,
held-out-family, modern-generator), TPR@1%FPR as the headline operating
point, calibration curve, Brier score. Segmentation/change-point
smoothing (spec 7C sequence-handling) and span-level IoU are deferred to
whatever time remains -- document-level evaluation only, given the
document-level model (see build_training_frame.py docstring, ADR 0008).
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

from detector.config import RedlineConfig, load_config
from detector.model.build_training_frame import Z_COLS, build_zscored_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.01) -> tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    raw_idx = int(np.searchsorted(fpr, target_fpr, side="right")) - 1
    idx: int = raw_idx if raw_idx > 0 else 0
    return float(tpr[idx]), float(thresholds[idx])


def run(config: RedlineConfig) -> dict:
    model_path = config.paths.features.parent / "model" / "logistic_regression.joblib"
    bundle = joblib.load(model_path)
    model = bundle["model"]

    frame = build_zscored_frame(config)
    frame_indexed = frame.reset_index(drop=False).rename(columns={"index": "orig_index"})
    test_rows = frame_indexed[frame_indexed["orig_index"].isin(bundle["test_index"])]

    X_test = test_rows[Z_COLS].fillna(0.0).values
    y_test = test_rows["label"].values
    scores_test = model.predict_proba(X_test)[:, 1]

    in_dist_tpr, threshold = tpr_at_fpr(y_test, scores_test)
    in_dist_auc = roc_auc_score(y_test, scores_test) if len(set(y_test)) > 1 else float("nan")
    brier = brier_score_loss(y_test, scores_test)

    human_mask = test_rows["label"] == 0
    human_X = test_rows[human_mask][Z_COLS].fillna(0.0).values
    human_scores = model.predict_proba(human_X)[:, 1] if len(human_X) else np.array([])

    results: dict = {
        "in_distribution": {
            "n": int(len(y_test)),
            "auc": float(in_dist_auc),
            "tpr_at_1pct_fpr": float(in_dist_tpr),
            "threshold_at_1pct_fpr": float(threshold),
            "brier_score": float(brier),
        }
    }

    for pool_name, cond_name in [("machine_heldout", "held_out_family"), ("modern_gen", "modern_generators")]:
        pool_rows = frame[frame["pool"] == pool_name]
        if pool_rows.empty or len(human_scores) == 0:
            results[cond_name] = {"n": 0, "note": "insufficient data"}
            continue
        pool_X = pool_rows[Z_COLS].fillna(0.0).values
        pool_scores = model.predict_proba(pool_X)[:, 1]

        combined_scores = np.concatenate([human_scores, pool_scores])
        combined_labels = np.concatenate([np.zeros(len(human_scores)), np.ones(len(pool_scores))])
        auc = roc_auc_score(combined_labels, combined_scores) if len(set(combined_labels)) > 1 else float("nan")
        tpr_at_fixed_threshold = float((pool_scores >= threshold).mean())

        results[cond_name] = {
            "n": int(len(pool_rows)),
            "auc": float(auc),
            "tpr_at_fixed_in_dist_threshold": tpr_at_fixed_threshold,
            "per_generator_breakdown": "omitted -- n too small per generator to support it (see EVALUATION.md)",
        }

    out_path = config.paths.features.parent / "model" / "evaluation_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Wrote evaluation results to %s", out_path)
    logger.info(json.dumps(results, indent=2))
    return results


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

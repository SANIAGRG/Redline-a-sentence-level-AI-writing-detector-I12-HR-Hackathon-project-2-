"""Module 7 support: bias audit (FPR by ELL status, C5) and the three
confidently-wrong essays (spec 7A item 9). Run after train.py/evaluate.py.
"""

from __future__ import annotations

import json
import logging
import math

import joblib
import numpy as np
import pandas as pd

from detector.config import RedlineConfig, load_config
from detector.model.build_training_frame import Z_COLS, build_zscored_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval -- more honest than a normal
    approximation at the tiny sample sizes this deadline run has."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return ((center - spread) / denom, (center + spread) / denom)


def run_bias_audit(config: RedlineConfig) -> dict:
    bundle = joblib.load(config.paths.features.parent / "model" / "logistic_regression.joblib")
    model = bundle["model"]

    frame = build_zscored_frame(config)
    frame_indexed = frame.reset_index(drop=False).rename(columns={"index": "orig_index"})
    test_rows = frame_indexed[frame_indexed["orig_index"].isin(bundle["test_index"])]
    human_test = test_rows[test_rows["label"] == 0]

    X = human_test[Z_COLS].fillna(0.0).values
    scores = model.predict_proba(X)[:, 1] if len(X) else np.array([])

    results_path = config.paths.features.parent / "model" / "evaluation_results.json"
    if results_path.exists():
        threshold = json.loads(results_path.read_text())["in_distribution"]["threshold_at_1pct_fpr"]
    else:
        logger.warning("No evaluation_results.json found -- falling back to threshold=0.5. Run evaluate.py first.")
        threshold = 0.5

    results: dict[str, object] = {}
    for status in ["Yes", "No"]:
        mask = human_test["ell_status"] == status
        n = int(mask.sum())
        if n == 0:
            results[status] = {"n": 0, "note": "no essays in this group at this pool size"}
            continue
        flagged = int((scores[mask.values] >= threshold).sum())
        fpr = flagged / n
        lo, hi = wilson_interval(flagged, n)
        results[status] = {"n": n, "fpr": fpr, "ci_95": [lo, hi], "flagged": flagged}

    note = (
        "Sample sizes here are tiny (deadline pool cut, ADR 0008) -- confidence intervals "
        "are correspondingly wide. Read the interval, not just the point estimate."
    )
    results["_note"] = note
    logger.info(json.dumps(results, indent=2, default=str))
    return results


def find_confident_failures(config: RedlineConfig, n: int = 3) -> list[dict]:
    """Searches every eval condition (in-distribution test split,
    held-out-family, modern-gen) for wrong predictions, not just the
    in-distribution split -- with perfect in-distribution accuracy at
    this pool size (see docs/EVALUATION.md), the instructive failures
    are in the conditions where TPR actually dropped.
    """
    bundle = joblib.load(config.paths.features.parent / "model" / "logistic_regression.joblib")
    model = bundle["model"]

    frame = build_zscored_frame(config)
    frame_indexed = frame.reset_index(drop=False).rename(columns={"index": "orig_index"})
    in_dist_test = frame_indexed[frame_indexed["orig_index"].isin(bundle["test_index"])]
    eval_pools = frame_indexed[frame_indexed["pool"].isin(["machine_heldout", "modern_gen"])]
    test_rows = pd.concat([in_dist_test, eval_pools]).drop_duplicates(subset="orig_index").copy()

    X = test_rows[Z_COLS].fillna(0.0).values
    test_rows["score"] = model.predict_proba(X)[:, 1]
    test_rows["pred"] = (test_rows["score"] >= 0.5).astype(int)
    test_rows["wrong"] = test_rows["pred"] != test_rows["label"]
    test_rows["confidence"] = (test_rows["score"] - 0.5).abs()

    wrong = test_rows[test_rows["wrong"]].sort_values("confidence", ascending=False)
    cases = []
    for _, row in wrong.head(n).iterrows():
        cases.append(
            {
                "doc_id": row["doc_id"],
                "pool": row["pool"],
                "true_label": "human" if row["label"] == 0 else "machine",
                "predicted_score": float(row["score"]),
                "ell_status": row.get("ell_status", "n/a"),
            }
        )
    return cases


def main() -> None:
    config = load_config()
    audit = run_bias_audit(config)
    failures = find_confident_failures(config)

    out_dir = config.paths.features.parent / "model"
    (out_dir / "bias_audit.json").write_text(json.dumps(audit, indent=2, default=str))
    (out_dir / "failure_cases.json").write_text(json.dumps(failures, indent=2, default=str))
    logger.info("Wrote bias_audit.json and failure_cases.json to %s", out_dir)


if __name__ == "__main__":
    main()

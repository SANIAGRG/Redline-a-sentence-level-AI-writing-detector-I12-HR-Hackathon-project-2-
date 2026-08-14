"""Module 7 support: ablation table (spec 7A item 10). Cheap to compute
retroactively -- retrains logistic regression on feature-family subsets
of the already-scored data, no new document scoring needed.
"""

from __future__ import annotations

import json
import logging

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from detector.config import RedlineConfig, load_config
from detector.model.build_training_frame import LIKELIHOOD_COLS, STYLOMETRIC_COLS, build_zscored_frame
from detector.model.train import TRAIN_POOLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_FAMILIES = {
    "likelihood_all": LIKELIHOOD_COLS,
    "cross_ppl_ratio": ["cross_ppl_ratio"],
    "punctuation_construction": [
        "em_dash_rate",
        "semicolon_rate",
        "oxford_comma_rate",
        "construction_template_rate",
    ],
    "error_rates": ["spelling_error_rate", "homophone_error_rate"],
    "concreteness": ["proper_noun_rate", "numeral_rate", "named_entity_rate"],
}


def run(config: RedlineConfig) -> dict:
    frame = build_zscored_frame(config)
    trainable = frame[frame["pool"].isin(TRAIN_POOLS)].copy()
    all_z_cols = [f"{c}_z" for c in STYLOMETRIC_COLS + LIKELIHOOD_COLS]

    results = {}

    def fit_and_score(cols: list[str], seed: int) -> float | None:
        X = trainable[cols].fillna(0.0).values
        y = trainable["label"].values
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
        except ValueError:
            return None
        lr = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=seed)
        try:
            model = CalibratedClassifierCV(lr, method="sigmoid", cv=2)
            model.fit(X_train, y_train)
        except Exception:
            return None
        scores = model.predict_proba(X_test)[:, 1]
        if len(set(y_test)) < 2:
            return None
        return float(roc_auc_score(y_test, scores))

    results["full_model"] = fit_and_score(all_z_cols, config.sampling.seed)

    for family_name, family_cols in FEATURE_FAMILIES.items():
        family_z_cols = [f"{c}_z" for c in family_cols]
        remaining_cols = [c for c in all_z_cols if c not in family_z_cols]
        results[f"without_{family_name}"] = fit_and_score(remaining_cols, config.sampling.seed)

    return results


def main() -> None:
    config = load_config()
    results = run(config)
    out_path = config.paths.features.parent / "model" / "ablation.json"
    out_path.write_text(json.dumps(results, indent=2))
    logger.info(json.dumps(results, indent=2))
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()

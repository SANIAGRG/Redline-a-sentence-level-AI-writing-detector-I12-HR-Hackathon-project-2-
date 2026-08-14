"""Adversarial degradation table (spec 7A item 11): compares the
trained model's score on each adversarial essay's original text vs. its
attacked (typo-injected or paraphrased) version. All 100 source essays
are genuinely machine-authored (Module 3C draws only from DAIGT machine
pools) -- a successful attack should visibly lower the score.
"""

from __future__ import annotations

import json
import logging

import joblib
import pandas as pd

from detector.config import RedlineConfig, load_config
from detector.features.corpus_relative import build_baseline_stats, length_band
from detector.model.build_training_frame import (
    ALL_FEATURE_COLS,
    build_zscored_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(config: RedlineConfig) -> dict:
    bundle = joblib.load(config.paths.features.parent / "model" / "logistic_regression.joblib")
    model = bundle["model"]

    checkpoint = pd.read_parquet(config.paths.features / "adversarial_features_checkpoint.parquet")
    stylo_rows = checkpoint["stylo_json"].map(json.loads)
    stylo_df = pd.json_normalize(stylo_rows)
    frame = pd.concat([checkpoint.drop(columns=["stylo_json"]).reset_index(drop=True), stylo_df], axis=1)
    frame["word_count"] = frame["n_words"]

    baseline_frame = build_zscored_frame(config)
    baseline = baseline_frame[baseline_frame["pool"] == "baseline"]
    baseline_stats = build_baseline_stats(baseline, ALL_FEATURE_COLS)
    baseline_means = baseline[ALL_FEATURE_COLS].mean()

    z_cols = []
    for col in ALL_FEATURE_COLS:
        z_col = f"{col}_z"
        z_cols.append(z_col)
        values = []
        for _, row in frame.iterrows():
            band = length_band(row["word_count"])
            mean_key, std_key = f"{col}__mean", f"{col}__std"
            if band in baseline_stats.index and std_key in baseline_stats.columns:
                mean = baseline_stats.loc[band, mean_key]
                std = baseline_stats.loc[band, std_key]
                if pd.isna(std) or std == 0:
                    mean, std = baseline_means[col], 1.0
            else:
                mean, std = baseline_means[col], 1.0
            values.append((row[col] - mean) / std if std else 0.0)
        frame[z_col] = values

    X = frame[z_cols].fillna(0.0).values
    frame["score"] = model.predict_proba(X)[:, 1]

    pivot = frame.pivot(index="essay_id", columns="version", values="score")
    attack_type = frame.drop_duplicates("essay_id").set_index("essay_id")["attack_type"]
    pivot["attack_type"] = attack_type
    pivot["degradation"] = pivot["original"] - pivot["attacked"]

    results: dict[str, object] = {
        "n": int(len(pivot)),
        "mean_original_score": float(pivot["original"].mean()),
        "mean_attacked_score": float(pivot["attacked"].mean()),
        "mean_degradation": float(pivot["degradation"].mean()),
    }
    for atk in ["typo", "paraphrase"]:
        sub = pivot[pivot["attack_type"] == atk]
        n_flipped = int(((sub["original"] >= 0.5) & (sub["attacked"] < 0.5)).sum()) if len(sub) else 0
        results[atk] = {
            "n": int(len(sub)),
            "mean_original_score": float(sub["original"].mean()),
            "mean_attacked_score": float(sub["attacked"].mean()),
            "mean_degradation": float(sub["degradation"].mean()),
            "n_flipped_below_0.5": n_flipped,
        }

    out_path = config.paths.features.parent / "model" / "adversarial_degradation.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(json.dumps(results, indent=2, default=str))
    return results


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

"""Joins stylometric (7B, document-level) and likelihood (7A, mean over
3 sampled sentences per document -- see ADR 0008 note below) into one
document-level feature frame, z-scored against the baseline pool (7C).

**Document-level, not sentence-level (further deadline simplification,
ADR 0008):** ADR 0002 makes the sentence the primary unit, but with only
3 sampled sentences/document at this pool size, a genuine sentence-level
model would have too few effective points per class to fit or evaluate
meaningfully tonight. Likelihood's 3 sampled sentences are mean-
aggregated to one row per document here; the underlying per-sentence
values are preserved in likelihood_checkpoint.parquet for anyone
extending this later. Documented in docs/LIMITATIONS.md.
"""

from __future__ import annotations

import pandas as pd

from detector.config import RedlineConfig
from detector.features.corpus_relative import build_baseline_stats, zscore_features

LIKELIHOOD_COLS = [
    "mean_logprob",
    "mean_lograsnk",
    "lrr",
    "mean_entropy",
    "mean_curvature",
    "logprob_variance",
    "cross_ppl_ratio",
]

STYLOMETRIC_COLS = [
    "sent_len_mean",
    "sent_len_std",
    "sent_len_burstiness",
    "mattr",
    "hapax_rate",
    "function_word_rate",
    "pos_trigram_entropy",
    "clause_count_mean",
    "subordination_ratio_mean",
    "em_dash_rate",
    "semicolon_rate",
    "oxford_comma_rate",
    "construction_template_rate",
    "discourse_marker_rate",
    "spelling_error_rate",
    "homophone_error_rate",
    "proper_noun_rate",
    "numeral_rate",
    "named_entity_rate",
]

ALL_FEATURE_COLS = STYLOMETRIC_COLS + LIKELIHOOD_COLS
Z_COLS = [f"{c}_z" for c in ALL_FEATURE_COLS]

# machine==1, human/mixed handled per-pool by caller
POOL_LABEL = {
    "baseline": 0,
    "human_training": 0,
    "machine_training": 1,
    "machine_heldout": 1,
    "modern_gen": 1,
    "polished": -1,  # mixed -- not a binary label, span-level eval only
}


def load_joined_frame(config: RedlineConfig) -> pd.DataFrame:
    stylo = pd.read_parquet(config.paths.features / "stylometric_all.parquet")
    likelihood = pd.read_parquet(config.paths.features / "likelihood_checkpoint.parquet")

    likelihood_doc = likelihood.groupby("doc_id")[LIKELIHOOD_COLS].mean().reset_index()

    merged = stylo.merge(likelihood_doc, on="doc_id", how="inner")
    merged["label"] = merged["pool"].map(POOL_LABEL)
    merged["word_count"] = merged["n_words"]  # stylometric.py names it n_words; z-scoring expects word_count

    # ell_status for the bias audit (C5) -- only meaningful for human rows,
    # sourced via doc_id == text_hash for baseline/human_training.
    joined = pd.read_parquet(config.paths.interim / "daigt_persuade_joined.parquet")
    ell_lookup = joined.drop_duplicates(subset="text_hash").set_index("text_hash")["ell_status"]
    merged["ell_status"] = merged["doc_id"].map(ell_lookup).fillna("n/a")
    return merged


def build_zscored_frame(config: RedlineConfig) -> pd.DataFrame:
    merged = load_joined_frame(config)
    baseline = merged[merged["pool"] == "baseline"]

    baseline_stats = build_baseline_stats(baseline, ALL_FEATURE_COLS)
    zscored = zscore_features(merged, ALL_FEATURE_COLS, baseline_stats)
    return zscored

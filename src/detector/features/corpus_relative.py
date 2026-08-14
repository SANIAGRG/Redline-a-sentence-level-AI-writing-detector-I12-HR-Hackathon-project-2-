"""Corpus-relative signals (Module 4, spec 7C, minus the embedding-
centroid feature -- cut under the submission deadline, see ADR 0008).

Z-scores every stylometric + likelihood feature against the human
baseline pool, conditioned on length band (short/medium/long by word
count) so a naturally-longer essay isn't flagged just for being long.
Also computes intra-document consistency: each paragraph's deviation
from its own document's median style vector -- the highest-value signal
for the polished/mixed case (ADR 0002), protected even under the
deadline crunch per explicit instruction.
"""

from __future__ import annotations

import pandas as pd

LENGTH_BANDS = [(0, 250, "short"), (250, 500, "medium"), (500, float("inf"), "long")]


def length_band(word_count: float) -> str:
    for lo, hi, name in LENGTH_BANDS:
        if lo <= word_count < hi:
            return name
    return "long"


_POOLED_BAND = "_all"  # sentinel row holding baseline-wide (band-agnostic) stats


def build_baseline_stats(baseline: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-feature mean/std, conditioned on length band, from the human
    baseline pool. `baseline` must have a `word_count` column. Also
    includes a `_all` row with baseline-wide pooled stats, used as the
    fallback when a band has too few baseline essays for a stable std --
    the fallback must come from the baseline population, not from
    whatever rows happen to be getting scored (see zscore_features).
    """
    df = baseline.copy()
    df["length_band"] = df["word_count"].map(length_band)
    stats = df.groupby("length_band")[feature_cols].agg(["mean", "std"])
    stats.columns = [f"{col}__{stat}" for col, stat in stats.columns]

    for col in feature_cols:
        stats.loc[_POOLED_BAND, f"{col}__mean"] = df[col].mean()
        stats.loc[_POOLED_BAND, f"{col}__std"] = df[col].std()
    return stats


def zscore_features(
    df: pd.DataFrame, feature_cols: list[str], baseline_stats: pd.DataFrame
) -> pd.DataFrame:
    """Z-score each row's features against the baseline stats for its
    own length band. Falls back to the baseline's pooled (all-band)
    mean/std -- never to statistics of `df` itself, which may be a
    single new document or a mixed human+machine batch, neither of
    which is a valid reference distribution.
    """
    out = df.copy()
    out["length_band"] = out["word_count"].map(length_band)

    for col in feature_cols:
        z_col = f"{col}_z"
        pooled_mean = baseline_stats.loc[_POOLED_BAND, f"{col}__mean"]
        pooled_std = baseline_stats.loc[_POOLED_BAND, f"{col}__std"]
        if pd.isna(pooled_std) or pooled_std == 0:
            pooled_std = 1.0

        values = []
        for _, row in out.iterrows():
            band = row["length_band"]
            mean_key, std_key = f"{col}__mean", f"{col}__std"
            if band in baseline_stats.index and std_key in baseline_stats.columns:
                mean = baseline_stats.loc[band, mean_key]
                std = baseline_stats.loc[band, std_key]
                if pd.isna(std) or std == 0:
                    mean, std = pooled_mean, pooled_std
            else:
                mean, std = pooled_mean, pooled_std
            values.append((row[col] - mean) / std if std else 0.0)
        out[z_col] = values
    return out


def intra_document_consistency(
    sentence_features: pd.DataFrame, feature_cols: list[str], doc_col: str = "doc_id"
) -> pd.DataFrame:
    """Each row's (paragraph-level, or sentence-level as given) deviation
    from its own document's median across feature_cols -- a single
    scalar per row: mean absolute deviation from the doc median,
    z-scored feature by feature so different units don't dominate.
    """
    df = sentence_features.copy()
    normed = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std().replace(0, 1.0)
    normed[doc_col] = df[doc_col].values
    doc_median = normed.groupby(doc_col)[feature_cols].transform("median")
    deviation = (normed[feature_cols] - doc_median).abs().mean(axis=1)
    df["intra_doc_consistency"] = deviation.values
    return df

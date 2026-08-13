"""Module 2 sampling: the four pools that come from data we already have
(DAIGT/PERSUADE join + raw DAIGT machine essays). The other three pools
(modern generators, polished/mixed, adversarial) are Module 3's job.

All four pools are disjoint by essay identity (text_hash for human rows,
DataFrame index for machine rows) so no essay is double-counted across
the baseline/training/held-out split.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from detector.config import RedlineConfig
from detector.ingest.loaders import load_daigt

# Excluded entirely from machine_training -- reserved for the held-out-family
# eval condition (Module 5). Falcon is architecturally distinct from every
# other DAIGT family (no sibling model shares its lineage, unlike the two
# Mistral or two Claude-derived families), and large enough (1,055 essays)
# to comfortably supply the held-out pool. See docs/DATA_CARD.md.
HELD_OUT_FAMILY = "falcon_180b_v1"

# Not a real generator family -- 3 stray mislabelled rows in DAIGT.
_EXCLUDED_SOURCES = {"persuade_corpus", "train_essays"}


@dataclass(frozen=True)
class SampledPools:
    human_baseline: pd.DataFrame
    human_training: pd.DataFrame
    machine_training: pd.DataFrame
    machine_heldout_family: pd.DataFrame


def sample_human_pools(
    joined: pd.DataFrame, baseline_n: int, training_n: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the joined human corpus into a baseline pool and an
    ELL-balanced (~50/50) training pool, disjoint from each other.
    """
    baseline = joined.sample(n=min(baseline_n, len(joined)), random_state=seed)
    remaining = joined.drop(baseline.index)

    ell = remaining["ell_status"].fillna("").str.strip()
    yes_pool = remaining[ell == "Yes"]
    no_pool = remaining[ell == "No"]

    half = training_n // 2
    n_yes = min(half, len(yes_pool))
    n_no = min(training_n - n_yes, len(no_pool))

    training = pd.concat(
        [
            yes_pool.sample(n=n_yes, random_state=seed),
            no_pool.sample(n=n_no, random_state=seed),
        ]
    )
    return baseline, training


def _stratify_by_topic(
    pool: pd.DataFrame, topic_share: pd.Series, target_n: int, seed: int
) -> pd.DataFrame:
    """Sample target_n rows from pool with per-topic quotas matching
    topic_share as closely as availability allows. Within a topic, rows
    are drawn uniformly at random -- if several families cover that
    topic, this naturally preserves whatever family mix exists for it.
    """
    quotas = (topic_share * target_n).round().astype(int)
    parts = []
    for topic, quota in quotas.items():
        topic_df = pool[pool["prompt_name"] == topic]
        n = min(quota, len(topic_df))
        if n > 0:
            parts.append(topic_df.sample(n=n, random_state=seed))
    sampled = pd.concat(parts) if parts else pool.iloc[0:0]
    if len(sampled) > target_n:
        sampled = sampled.sample(n=target_n, random_state=seed)
    return sampled


def sample_machine_pools(
    daigt: pd.DataFrame,
    human_topic_share: pd.Series,
    training_n: int,
    heldout_n: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample the machine training pool and the held-out-family pool,
    both stratified by topic to match PERSUADE's (human) topic
    distribution -- DAIGT's own machine essays are heavily topic-skewed
    (some prompts 10x overrepresented vs. others; see docs/DATA_CARD.md),
    so plain family-proportional sampling would inherit that skew and
    let the classifier learn topic instead of authorship.
    """
    machine = daigt[(daigt["label"] == 1) & (~daigt["source"].isin(_EXCLUDED_SOURCES))]

    heldout_family = machine[machine["source"] == HELD_OUT_FAMILY]
    heldout = _stratify_by_topic(heldout_family, human_topic_share, heldout_n, seed)

    trainable = machine[machine["source"] != HELD_OUT_FAMILY]
    training = _stratify_by_topic(trainable, human_topic_share, training_n, seed)

    return training, heldout


def build_sampled_pools(config: RedlineConfig) -> SampledPools:
    joined = pd.read_parquet(config.paths.interim / "daigt_persuade_joined.parquet")
    daigt = load_daigt(config.paths.raw)

    pool_counts = {p.name: p.count for p in config.sampling.pools}
    seed = config.sampling.seed

    human_baseline, human_training = sample_human_pools(
        joined, pool_counts["human_baseline"], pool_counts["human_training"], seed
    )

    # Population-level topic distribution (all 25,996 PERSUADE essays, not
    # just the sampled baseline) is the reference the machine pools are
    # stratified against.
    human_topic_share = joined["prompt_name"].value_counts(normalize=True)
    machine_training, machine_heldout = sample_machine_pools(
        daigt,
        human_topic_share,
        pool_counts["machine_training"],
        pool_counts["machine_heldout_family"],
        seed,
    )

    return SampledPools(
        human_baseline=human_baseline,
        human_training=human_training,
        machine_training=machine_training,
        machine_heldout_family=machine_heldout,
    )

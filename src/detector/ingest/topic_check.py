"""Topic-distribution check (Module 2 exit criterion).

DAIGT's machine essays were generated from PERSUADE prompts, so human and
machine topic coverage should already align. If they don't, the
classifier would learn topic rather than authorship. This compares the
two distributions and flags any prompt with a large gap.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# A gap larger than this between human-share and machine-share for any
# single topic is flagged as a potential leakage risk.
MAX_ACCEPTABLE_GAP_PCT = 5.0


def topic_distribution_comparison(manifest: pd.DataFrame) -> pd.DataFrame:
    human = manifest[manifest["author_type"] == "human"]
    machine = manifest[manifest["author_type"] == "machine"]

    human_pct = human["topic"].value_counts(normalize=True).mul(100)
    machine_pct = machine["topic"].value_counts(normalize=True).mul(100)

    comparison = pd.DataFrame({"human_pct": human_pct, "machine_pct": machine_pct}).fillna(0.0)
    comparison["gap_pct"] = (comparison["human_pct"] - comparison["machine_pct"]).abs()
    return comparison.sort_values("gap_pct", ascending=False)


def plot_topic_distribution(comparison: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    comparison[["human_pct", "machine_pct"]].plot(kind="barh", ax=ax)
    ax.set_xlabel("% of pool")
    ax.set_title("Topic distribution: human vs machine")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

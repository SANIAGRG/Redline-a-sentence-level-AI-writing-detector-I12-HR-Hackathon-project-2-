"""Loaders for the two user-provided raw datasets.

`dataset/raw/` is read-only — these functions only read from it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_daigt(raw_dir: Path) -> pd.DataFrame:
    """Load DAIGT-V2 (train_v2_drcat_02.csv).

    Columns: text, label, prompt_name, source, RDizzl3_seven.
    label 0 = human, 1 = machine. source == "persuade_corpus" identifies
    the human rows that are literal PERSUADE essays with demographics
    stripped.
    """
    return pd.read_csv(raw_dir / "train_v2_drcat_02.csv")


def load_persuade(raw_dir: Path) -> pd.DataFrame:
    """Load PERSUADE 2.0 with demographics (persuade_2.0_human_scores_demo_id_github.csv).

    Columns include full_text, ell_status, gender, grade_level,
    race_ethnicity, economically_disadvantaged, student_disability_status.
    """
    return pd.read_csv(raw_dir / "persuade_2.0_human_scores_demo_id_github.csv")

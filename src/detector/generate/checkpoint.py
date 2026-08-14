"""Shared resumable-checkpoint pattern (H4): append one row to a parquet
file at a time via a write-to-tmp-then-atomic-replace, so a run killed
mid-way leaves the checkpoint in a valid, resumable state rather than a
half-written file. Used by every Module 3 generation runner.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_checkpoint(checkpoint_path: Path, columns: list[str]) -> pd.DataFrame:
    if checkpoint_path.exists():
        return pd.read_parquet(checkpoint_path)
    return pd.DataFrame(columns=columns)


def append_checkpoint(checkpoint_path: Path, row: dict, columns: list[str]) -> None:
    existing = load_checkpoint(checkpoint_path, columns)
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    tmp_path = checkpoint_path.with_suffix(".parquet.tmp")
    updated.to_parquet(tmp_path, index=False)
    tmp_path.replace(checkpoint_path)

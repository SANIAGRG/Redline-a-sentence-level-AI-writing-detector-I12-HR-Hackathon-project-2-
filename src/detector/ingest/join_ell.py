"""The ELL join: attach PERSUADE demographics to DAIGT's human rows.

DAIGT-V2's human rows (source == "persuade_corpus") are PERSUADE essays
with demographic columns dropped. PERSUADE 2.0 has those columns. The two
releases do not share IDs, so we join on normalised-text hash.

Run directly to print the match-rate and ELL-count report used for the
Module 1 exit criteria. `make join` calls this as a script.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from detector.config import load_config
from detector.ingest.loaders import load_daigt, load_persuade
from detector.ingest.normalisation import text_hash

ELL_MIN_USABLE = 200


@dataclass(frozen=True)
class JoinReport:
    daigt_persuade_rows: int
    persuade_rows: int
    matched_rows: int
    match_rate: float
    ell_yes_matched: int
    ell_no_matched: int
    ell_unknown_matched: int
    recommend_ellipse: bool

    def summary(self) -> str:
        lines = [
            f"DAIGT persuade_corpus rows:  {self.daigt_persuade_rows}",
            f"PERSUADE rows:               {self.persuade_rows}",
            f"Matched rows:                {self.matched_rows}",
            f"Match rate:                  {self.match_rate:.1%}",
            f"  ELL='Yes' (usable):        {self.ell_yes_matched}",
            f"  ELL='No':                  {self.ell_no_matched}",
            f"  ELL unknown/missing:       {self.ell_unknown_matched}",
        ]
        if self.recommend_ellipse:
            lines.append(
                f"\nRECOMMENDATION: usable ELL essay count ({self.ell_yes_matched}) is "
                f"below the ~{ELL_MIN_USABLE} threshold. Download ELLIPSE "
                "(https://github.com/scrosseye/ELLIPSE-Corpus, Code -> Download ZIP) "
                "and place it in dataset/raw/ before continuing to Module 2."
            )
        else:
            lines.append(
                f"\nRECOMMENDATION: usable ELL essay count ({self.ell_yes_matched}) is "
                f"comfortably above the ~{ELL_MIN_USABLE} threshold. Proceeding without "
                "ELLIPSE."
            )
        return "\n".join(lines)


def build_joined_manifest(raw_dir: Path) -> tuple[pd.DataFrame, JoinReport]:
    daigt = load_daigt(raw_dir)
    persuade = load_persuade(raw_dir)

    daigt_human = daigt[daigt["source"] == "persuade_corpus"].copy()
    daigt_human["text_hash"] = daigt_human["text"].map(text_hash)
    persuade = persuade.copy()
    persuade["text_hash"] = persuade["full_text"].map(text_hash)

    # A handful of essays are byte-identical across students in PERSUADE
    # (e.g. boilerplate/empty submissions); keep first occurrence so the
    # join stays 1:1 and does not fan out.
    persuade_dedup = persuade.drop_duplicates(subset="text_hash", keep="first")

    merged = daigt_human.merge(
        persuade_dedup,
        on="text_hash",
        how="left",
        suffixes=("", "_persuade"),
        indicator=True,
    )

    matched = merged[merged["_merge"] == "both"].copy()
    match_rate = len(matched) / len(daigt_human) if len(daigt_human) else 0.0

    ell = matched["ell_status"].fillna("").str.strip()
    ell_yes = int((ell == "Yes").sum())
    ell_no = int((ell == "No").sum())
    ell_unknown = int(len(matched) - ell_yes - ell_no)

    report = JoinReport(
        daigt_persuade_rows=len(daigt_human),
        persuade_rows=len(persuade),
        matched_rows=len(matched),
        match_rate=match_rate,
        ell_yes_matched=ell_yes,
        ell_no_matched=ell_no,
        ell_unknown_matched=ell_unknown,
        recommend_ellipse=ell_yes < ELL_MIN_USABLE,
    )
    return matched, report


def main() -> int:
    config = load_config()
    raw_dir = config.paths.raw
    interim_dir = config.paths.interim
    interim_dir.mkdir(parents=True, exist_ok=True)

    matched, report = build_joined_manifest(raw_dir)

    out_path = interim_dir / "daigt_persuade_joined.parquet"
    matched.drop(columns=["_merge"]).to_parquet(out_path, index=False)

    print(report.summary())
    print(f"\nWrote {len(matched)} joined rows to {out_path}")

    if report.match_rate < 0.80:
        print(
            "\nWARNING: match rate below the ~80% workable threshold set in the spec.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

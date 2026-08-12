"""Resumable polish/mixed corpus runner (Module 3A, kicked off in Module 1).

Takes human essays and has llama3.2:3b revise a random subset of each
essay's paragraphs at one of three intensities (grammar-fix, flow
improvement, full rewrite), leaving the rest of the essay untouched. This
produces the "mixed" corpus: a human draft with some paragraphs
model-polished and some not, with per-paragraph ground truth on which is
which — the case the brief calls realistic (see ADR 0002).

Checkpointing (H4): after every essay, the row (with per-paragraph
original/revised text and a `was_revised` flag as JSON) is appended to
`dataset/generated/polish/checkpoint.parquet`. On restart, already
completed essay ids are read from the checkpoint and skipped, so a run
killed mid-way resumes rather than restarting from zero.

`make generate` runs this. Runs standalone: `python -m detector.generate.run_polish`.
"""

from __future__ import annotations

import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from detector.config import RedlineConfig, load_config
from detector.generate.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLISH_MODEL = "llama3.2:3b"
INTENSITIES = ("grammar", "flow", "rewrite")
MIN_WORD_COUNT = 200  # need multiple real paragraphs to make a mixed doc meaningful
MIN_PARAGRAPHS = 2

PROMPTS: dict[str, str] = {
    "grammar": (
        "Fix only grammar, spelling, and punctuation errors in the following paragraph. "
        "Do not change wording, tone, or sentence structure otherwise. "
        "Return only the corrected paragraph, nothing else.\n\nParagraph:\n{paragraph}"
    ),
    "flow": (
        "Improve the flow and sentence transitions of the following paragraph while "
        "keeping the same content, meaning, and approximate length. "
        "Return only the revised paragraph, nothing else.\n\nParagraph:\n{paragraph}"
    ),
    "rewrite": (
        "Rewrite the following paragraph, preserving its content and meaning but "
        "freely changing wording and sentence structure. "
        "Return only the rewritten paragraph, nothing else.\n\nParagraph:\n{paragraph}"
    ),
}


@dataclass(frozen=True)
class ParagraphResult:
    paragraph_index: int
    original_text: str
    revised_text: str
    was_revised: bool


def split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n")]
    return [p for p in paragraphs if p]


def select_essays(joined: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    eligible = joined[joined["word_count"] >= MIN_WORD_COUNT].copy()
    eligible = eligible[eligible["text"].map(lambda t: len(split_paragraphs(t)) >= MIN_PARAGRAPHS)]
    n = min(n, len(eligible))
    return eligible.sample(n=n, random_state=seed).reset_index(drop=True)


def assign_intensities(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    base = list(INTENSITIES) * (n // len(INTENSITIES) + 1)
    rng.shuffle(base)
    return base[:n]


def polish_essay(
    client: OllamaClient, text: str, intensity: str, seed: int
) -> list[ParagraphResult]:
    paragraphs = split_paragraphs(text)
    rng = random.Random(seed)
    # Revise a random subset of paragraphs (40-100%) so the doc stays "mixed"
    # rather than a full-essay rewrite, per Module 3A / ADR 0002.
    n_to_revise = max(1, round(len(paragraphs) * rng.uniform(0.4, 1.0)))
    revise_idx = set(rng.sample(range(len(paragraphs)), n_to_revise))

    results = []
    for i, para in enumerate(paragraphs):
        if i in revise_idx:
            prompt = PROMPTS[intensity].format(paragraph=para)
            revised = client.generate(POLISH_MODEL, prompt)
            results.append(ParagraphResult(i, para, revised, was_revised=True))
        else:
            results.append(ParagraphResult(i, para, para, was_revised=False))
    return results


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    if checkpoint_path.exists():
        return pd.read_parquet(checkpoint_path)
    return pd.DataFrame(
        columns=[
            "essay_id",
            "intensity",
            "n_paragraphs",
            "n_revised",
            "paragraphs_json",
        ]
    )


def append_checkpoint(checkpoint_path: Path, row: dict) -> None:
    existing = load_checkpoint(checkpoint_path)
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    tmp_path = checkpoint_path.with_suffix(".parquet.tmp")
    updated.to_parquet(tmp_path, index=False)
    tmp_path.replace(checkpoint_path)


def run(config: RedlineConfig, target_n: int | None = None) -> None:
    joined_path = config.paths.interim / "daigt_persuade_joined.parquet"
    if not joined_path.exists():
        logger.error("Run `make join` first — %s not found.", joined_path)
        sys.exit(1)

    joined = pd.read_parquet(joined_path)
    pool_count = next(
        (p.count for p in config.sampling.pools if p.name == "polished_mixed"), 350
    )
    n = target_n or pool_count

    out_dir = config.paths.generated / "polish"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.parquet"

    selected = select_essays(joined, n, seed=config.sampling.seed)
    intensities = assign_intensities(len(selected), seed=config.sampling.seed)
    selected["essay_id"] = selected["text_hash"]
    selected["intensity"] = intensities

    done = set(load_checkpoint(checkpoint_path)["essay_id"])
    remaining = selected[~selected["essay_id"].isin(done)]
    logger.info(
        "Polish run: %d target, %d already done, %d remaining.",
        len(selected),
        len(done),
        len(remaining),
    )

    client = OllamaClient(config.ollama)

    for i, (_, row) in enumerate(remaining.iterrows(), start=1):
        try:
            results = polish_essay(
                client, row["text"], row["intensity"], seed=hash(row["essay_id"]) % (2**31)
            )
        except Exception:
            logger.exception(
                "Failed on essay_id=%s — leaving unchecked, will retry on next run.",
                row["essay_id"],
            )
            continue

        checkpoint_row = {
            "essay_id": row["essay_id"],
            "intensity": row["intensity"],
            "n_paragraphs": len(results),
            "n_revised": sum(r.was_revised for r in results),
            "paragraphs_json": json.dumps(
                [
                    {
                        "paragraph_index": r.paragraph_index,
                        "original_text": r.original_text,
                        "revised_text": r.revised_text,
                        "was_revised": r.was_revised,
                    }
                    for r in results
                ]
            ),
        }
        append_checkpoint(checkpoint_path, checkpoint_row)
        logger.info(
            "[%d/%d] polished essay_id=%s intensity=%s (%d/%d paragraphs revised)",
            i,
            len(remaining),
            row["essay_id"][:12],
            row["intensity"],
            checkpoint_row["n_revised"],
            checkpoint_row["n_paragraphs"],
        )

    n_total = len(load_checkpoint(checkpoint_path))
    logger.info("Polish run complete. %d essays in checkpoint.", n_total)


def main() -> None:
    config = load_config()
    run(config)


if __name__ == "__main__":
    main()

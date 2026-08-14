"""Resumable modern-generator slice runner (Module 3B, the headline
temporal-generalization experiment -- ADR 0007).

Generates 50 essays each from three current-generation models
(llama3.2:3b, gemma2:2b, phi3.5:3.8b) -- none of which share lineage
with any DAIGT-V2 generator, all of which postdate DAIGT-V2's late-2023
construction. Uses PERSUADE's 8 self-contained ("Independent" task)
prompts, cycled for topic diversity, across four prompt conditions:
bare, persona, style-exemplar, evasion.

This slice is held out entirely for Module 5 evaluation -- never
training data.

Checkpointing (H4): same append-after-every-essay pattern as
run_polish.py (src/detector/generate/checkpoint.py). `make generate`
runs this after run_polish. Standalone:
`python -m detector.generate.run_modern`.
"""

from __future__ import annotations

import logging
import random
import sys

import pandas as pd

from detector.config import RedlineConfig, load_config
from detector.generate.checkpoint import append_checkpoint, load_checkpoint
from detector.generate.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS = ("llama3.2:3b", "gemma2:2b", "phi3.5:3.8b")
CONDITIONS = ("bare", "persona", "style_exemplar", "evasion")
N_PER_MODEL = 50

# Self-contained prompts only (PERSUADE task == "Independent") -- the other
# 7 prompts are "Text dependent" and reference a source passage; excluded
# from this module's scope, see ADR 0007.
INDEPENDENT_PROMPTS = (
    "Phones and driving",
    "Summer projects",
    "Mandatory extracurricular activities",
    "Community service",
    "Grades for extracurricular activities",
    "Cell phones at school",
    "Distance learning",
    "Seeking multiple opinions",
)

LENGTH_INSTRUCTION = (
    "Write a persuasive essay of about 350-450 words. "
    "Return only the essay text, nothing else -- no title, no preamble, no notes."
)

CHECKPOINT_COLUMNS = ["essay_id", "model", "condition", "topic", "text", "word_count"]


def build_prompt(assignment: str, condition: str, exemplar_text: str | None) -> str:
    if condition == "bare":
        return f"{assignment}\n\n{LENGTH_INSTRUCTION}"
    if condition == "persona":
        return (
            "You are a 17-year-old high school student applying to college, writing "
            f"a persuasive essay for a class assignment.\n\n{assignment}\n\n{LENGTH_INSTRUCTION}"
        )
    if condition == "style_exemplar":
        assert exemplar_text is not None
        return (
            "Here is an example of a well-written student essay:\n\n"
            f"{exemplar_text}\n\n"
            f"Now, in a similar style, respond to the following:\n\n{assignment}\n\n"
            f"{LENGTH_INSTRUCTION}"
        )
    if condition == "evasion":
        return (
            f"{assignment}\n\nWrite naturally, vary your sentence length, and avoid "
            f"sounding AI-generated.\n\n{LENGTH_INSTRUCTION}"
        )
    raise ValueError(f"unknown condition: {condition}")


def build_targets(seed: int) -> list[dict]:
    """Deterministic (model, condition, topic) assignment for all
    3 x N_PER_MODEL essays, evenly cycling conditions and topics.
    """
    targets = []
    for model in MODELS:
        for i in range(N_PER_MODEL):
            condition = CONDITIONS[i % len(CONDITIONS)]
            topic = INDEPENDENT_PROMPTS[i % len(INDEPENDENT_PROMPTS)]
            model_slug = model.replace(":", "-").replace(".", "-")
            targets.append(
                {
                    "essay_id": f"{model_slug}_{i:03d}",
                    "model": model,
                    "condition": condition,
                    "topic": topic,
                }
            )
    rng = random.Random(seed)
    rng.shuffle(targets)
    return targets


def pick_exemplar(joined: pd.DataFrame, topic: str, seed: int) -> str:
    """A short excerpt of a real human essay on the same topic, used as
    the style_exemplar condition's reference text.
    """
    candidates = joined[joined["prompt_name"] == topic]
    if candidates.empty:
        candidates = joined
    row = candidates.sample(n=1, random_state=seed).iloc[0]
    words = str(row["text"]).split()
    return " ".join(words[:150])


def run(config: RedlineConfig) -> None:
    joined_path = config.paths.interim / "daigt_persuade_joined.parquet"
    if not joined_path.exists():
        logger.error("Run `make join` first — %s not found.", joined_path)
        sys.exit(1)
    joined = pd.read_parquet(joined_path)

    out_dir = config.paths.generated / "modern"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "checkpoint.parquet"

    assignments = (
        joined[joined["prompt_name"].isin(INDEPENDENT_PROMPTS)]
        .drop_duplicates(subset="prompt_name")
        .set_index("prompt_name")["assignment"]
        .to_dict()
    )

    targets = build_targets(seed=config.sampling.seed)
    done = set(load_checkpoint(checkpoint_path, CHECKPOINT_COLUMNS)["essay_id"])
    remaining = [t for t in targets if t["essay_id"] not in done]
    logger.info(
        "Modern-gen run: %d target, %d already done, %d remaining.",
        len(targets),
        len(done),
        len(remaining),
    )

    client = OllamaClient(config.ollama)

    for i, target in enumerate(remaining, start=1):
        exemplar = None
        if target["condition"] == "style_exemplar":
            exemplar = pick_exemplar(
                joined, target["topic"], seed=hash(target["essay_id"]) % (2**31)
            )
        prompt = build_prompt(assignments[target["topic"]], target["condition"], exemplar)

        try:
            text = client.generate(target["model"], prompt)
        except Exception:
            logger.exception(
                "Failed on essay_id=%s — leaving unchecked, will retry on next run.",
                target["essay_id"],
            )
            continue

        checkpoint_row = {
            "essay_id": target["essay_id"],
            "model": target["model"],
            "condition": target["condition"],
            "topic": target["topic"],
            "text": text,
            "word_count": len(text.split()),
        }
        append_checkpoint(checkpoint_path, checkpoint_row, CHECKPOINT_COLUMNS)
        logger.info(
            "[%d/%d] generated essay_id=%s model=%s condition=%s (%d words)",
            i,
            len(remaining),
            target["essay_id"],
            target["model"],
            target["condition"],
            checkpoint_row["word_count"],
        )

    n_total = len(load_checkpoint(checkpoint_path, CHECKPOINT_COLUMNS))
    logger.info("Modern-gen run complete. %d essays in checkpoint.", n_total)


def main() -> None:
    run(load_config())


if __name__ == "__main__":
    main()

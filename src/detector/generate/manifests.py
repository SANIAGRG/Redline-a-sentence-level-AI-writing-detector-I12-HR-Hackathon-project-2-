"""Manifests for the three Module 3 generated sets (polish, modern-gen,
adversarial), matching the same schema as Module 2's DAIGT/PERSUADE
manifest (source, author_type, generator, prompt_condition, ell_flag,
topic, word_count, license) so downstream code can treat all seven pools
uniformly. Written alongside each set's checkpoint.

`author_type` gains one new value here beyond Module 2's "human"/
"machine": **"mixed"**, for the polish corpus -- a document that is
genuinely part human, part model-revised, per paragraph. See ADR 0002.
"""

from __future__ import annotations

import json

import pandas as pd

from detector.config import RedlineConfig
from detector.ingest.manifest import DAIGT_LICENSE, GENERATOR_NAMES, PERSUADE_LICENSE

POLISH_GENERATOR = "llama3.2:3b"
OLLAMA_LICENSE = "N/A -- generated locally via Ollama, not sourced from a licensed dataset"


def build_polish_manifest(config: RedlineConfig) -> pd.DataFrame:
    checkpoint_path = config.paths.generated / "polish" / "checkpoint.parquet"
    checkpoint = pd.read_parquet(checkpoint_path)

    joined = pd.read_parquet(config.paths.interim / "daigt_persuade_joined.parquet")
    # text_hash is unique in practice but not guaranteed -- 2 of 25,996 rows
    # collide (genuine duplicate essays in DAIGT's raw data). Keep first so
    # the metadata lookup below can't raise on a duplicate index.
    joined = joined.drop_duplicates(subset="text_hash", keep="first")
    meta = joined.set_index("text_hash")[["prompt_name", "word_count", "ell_status"]]
    meta = meta.reindex(checkpoint["essay_id"])

    ell = meta["ell_status"].fillna("").astype(str).str.strip().replace("", "unknown")
    word_count = checkpoint["paragraphs_json"].map(
        lambda pj: sum(len(p["revised_text"].split()) for p in json.loads(pj))
    )

    return pd.DataFrame(
        {
            "doc_id": checkpoint["essay_id"].values,
            "pool": "polished_mixed",
            "source": "persuade_corpus+llama3.2:3b",
            "author_type": "mixed",
            "generator": POLISH_GENERATOR,
            "prompt_condition": checkpoint["intensity"].values,
            "ell_flag": ell.values,
            "topic": meta["prompt_name"].values,
            "word_count": word_count.values,
            "license": PERSUADE_LICENSE,
        }
    )


def build_modern_manifest(config: RedlineConfig) -> pd.DataFrame:
    checkpoint_path = config.paths.generated / "modern" / "checkpoint.parquet"
    checkpoint = pd.read_parquet(checkpoint_path)

    return pd.DataFrame(
        {
            "doc_id": checkpoint["essay_id"].values,
            "pool": "machine_modern",
            "source": "ollama_modern_generated",
            "author_type": "machine",
            "generator": checkpoint["model"].values,
            "prompt_condition": checkpoint["condition"].values,
            "ell_flag": "n/a",
            "topic": checkpoint["topic"].values,
            "word_count": checkpoint["word_count"].values,
            "license": OLLAMA_LICENSE,
        }
    )


def build_adversarial_manifest(config: RedlineConfig) -> pd.DataFrame:
    checkpoint_path = config.paths.generated / "adversarial" / "checkpoint.parquet"
    checkpoint = pd.read_parquet(checkpoint_path)

    generator = checkpoint["source_family"].map(lambda s: GENERATOR_NAMES.get(s, s))
    generator = generator + " + " + checkpoint["attack_type"]

    return pd.DataFrame(
        {
            "doc_id": checkpoint["essay_id"].values,
            "pool": "adversarial",
            "source": checkpoint["source_family"].values,
            "author_type": "machine",
            "generator": generator.values,
            "prompt_condition": checkpoint["attack_type"].values,
            "ell_flag": "n/a",
            "topic": checkpoint["topic"].values,
            "word_count": checkpoint["word_count"].values,
            "license": DAIGT_LICENSE,
        }
    )

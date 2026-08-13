"""Manifest schema (Module 2): one row per document across every sampled
pool. Columns: source, author_type, generator, prompt_condition,
ell_flag, topic, word_count, license, plus a pool tag and doc_id used to
join against segmentation/feature tables downstream.

Licenses: PERSUADE 2.0 is confirmed CC BY-NC-SA 4.0. DAIGT-V2's license
could not be verified programmatically (Kaggle's page is JS-rendered);
confirmed usable directly by the project owner on the Kaggle page. See
docs/DATA_CARD.md.
"""

from __future__ import annotations

import pandas as pd

from detector.ingest.sampling import SampledPools

PERSUADE_LICENSE = "PERSUADE 2.0: CC BY-NC-SA 4.0"
DAIGT_LICENSE = "DAIGT-V2 (Kaggle): confirmed usable -- see DATA_CARD.md"

# source -> human-readable generator name, for manifest readability.
_GENERATOR_NAMES: dict[str, str] = {
    "mistral7binstruct_v2": "Mistral-7B-Instruct v2",
    "mistral7binstruct_v1": "Mistral-7B-Instruct v1",
    "mistralai/Mistral-7B-Instruct-v0.1": "Mistral-7B-Instruct v0.1",
    "chat_gpt_moth": "GPT-3.5 (chat_gpt_moth)",
    "llama2_chat": "LLaMA-2-Chat",
    "llama_70b_v1": "LLaMA-70B",
    "NousResearch/Llama-2-7b-chat-hf": "LLaMA-2-7B-Chat (NousResearch)",
    "kingki19_palm": "PaLM (kingki19_palm)",
    "palm-text-bison1": "PaLM Text-Bison-1",
    "falcon_180b_v1": "Falcon-180B",
    "darragh_claude_v6": "Claude v6 (darragh)",
    "darragh_claude_v7": "Claude v7 (darragh)",
    "radek_500": "GPT-3.5 (radek_500)",
    "radekgpt4": "GPT-4 (radek)",
    "cohere-command": "Cohere Command",
}


def _human_rows(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    ell = df["ell_status"].fillna("").str.strip().replace("", "unknown")
    return pd.DataFrame(
        {
            "doc_id": df["text_hash"],
            "pool": pool,
            "source": "persuade_corpus",
            "author_type": "human",
            "generator": "n/a",
            "prompt_condition": "n/a",
            "ell_flag": ell.values,
            "topic": df["prompt_name"].values,
            "word_count": df["word_count"].values,
            "license": PERSUADE_LICENSE,
        }
    )


def _machine_rows(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    doc_id = df["source"].astype(str) + "::" + df.index.astype(str)
    word_count = df["text"].map(lambda t: len(str(t).split()))
    generator = df["source"].map(lambda s: _GENERATOR_NAMES.get(s, s))
    return pd.DataFrame(
        {
            "doc_id": doc_id.values,
            "pool": pool,
            "source": df["source"].values,
            "author_type": "machine",
            "generator": generator.values,
            "prompt_condition": "unknown (DAIGT-provided)",
            "ell_flag": "n/a",
            "topic": df["prompt_name"].values,
            "word_count": word_count.values,
            "license": DAIGT_LICENSE,
        }
    )


def build_manifest(pools: SampledPools) -> pd.DataFrame:
    parts = [
        _human_rows(pools.human_baseline, "human_baseline"),
        _human_rows(pools.human_training, "human_training"),
        _machine_rows(pools.machine_training, "machine_training"),
        _machine_rows(pools.machine_heldout_family, "machine_heldout_family"),
    ]
    manifest = pd.concat(parts, ignore_index=True)
    assert manifest["doc_id"].is_unique, "manifest doc_id collision across pools"
    return manifest

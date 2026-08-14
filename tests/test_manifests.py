import json
from pathlib import Path

import pandas as pd
import pytest

from detector.config import RedlineConfig
from detector.generate.manifests import (
    build_adversarial_manifest,
    build_modern_manifest,
    build_polish_manifest,
)


@pytest.fixture
def config(tmp_path: Path) -> RedlineConfig:
    cfg = RedlineConfig()
    cfg.paths.interim = tmp_path / "interim"
    cfg.paths.generated = tmp_path / "generated"
    cfg.paths.interim.mkdir(parents=True)
    cfg.paths.generated.mkdir(parents=True)
    return cfg


def test_build_polish_manifest(config: RedlineConfig) -> None:
    joined = pd.DataFrame(
        {
            "text_hash": ["h1", "h2"],
            "prompt_name": ["Topic A", "Topic B"],
            "word_count": [300, 400],
            "ell_status": ["Yes", None],
        }
    )
    joined.to_parquet(config.paths.interim / "daigt_persuade_joined.parquet", index=False)

    polish_dir = config.paths.generated / "polish"
    polish_dir.mkdir(parents=True)
    checkpoint = pd.DataFrame(
        {
            "essay_id": ["h1", "h2"],
            "intensity": ["grammar", "rewrite"],
            "n_paragraphs": [2, 2],
            "n_revised": [1, 2],
            "paragraphs_json": [
                json.dumps(
                    [
                        {"paragraph_index": 0, "revised_text": "one two three"},
                        {"paragraph_index": 1, "revised_text": "four five"},
                    ]
                ),
                json.dumps(
                    [
                        {"paragraph_index": 0, "revised_text": "a b c d"},
                        {"paragraph_index": 1, "revised_text": "e f"},
                    ]
                ),
            ],
        }
    )
    checkpoint.to_parquet(polish_dir / "checkpoint.parquet", index=False)

    manifest = build_polish_manifest(config)

    assert len(manifest) == 2
    assert set(manifest["author_type"]) == {"mixed"}
    assert manifest.loc[manifest["doc_id"] == "h1", "ell_flag"].iloc[0] == "Yes"
    assert manifest.loc[manifest["doc_id"] == "h2", "ell_flag"].iloc[0] == "unknown"
    assert manifest.loc[manifest["doc_id"] == "h1", "word_count"].iloc[0] == 5


def test_build_polish_manifest_survives_duplicate_text_hash(config: RedlineConfig) -> None:
    # DAIGT has a small number of genuinely duplicate essays (2 of 25,996
    # in the real data) -- a naive set_index("text_hash") would raise on
    # reindex. Confirm the dedup fix keeps this from crashing.
    joined = pd.DataFrame(
        {
            "text_hash": ["h1", "h1", "h2"],
            "prompt_name": ["Topic A", "Topic A", "Topic B"],
            "word_count": [300, 300, 400],
            "ell_status": ["Yes", "Yes", None],
        }
    )
    joined.to_parquet(config.paths.interim / "daigt_persuade_joined.parquet", index=False)

    polish_dir = config.paths.generated / "polish"
    polish_dir.mkdir(parents=True)
    checkpoint = pd.DataFrame(
        {
            "essay_id": ["h1"],
            "intensity": ["grammar"],
            "n_paragraphs": [1],
            "n_revised": [1],
            "paragraphs_json": [
                json.dumps([{"paragraph_index": 0, "revised_text": "one two three"}])
            ],
        }
    )
    checkpoint.to_parquet(polish_dir / "checkpoint.parquet", index=False)

    manifest = build_polish_manifest(config)

    assert len(manifest) == 1
    assert manifest.iloc[0]["topic"] == "Topic A"


def test_build_modern_manifest(config: RedlineConfig) -> None:
    modern_dir = config.paths.generated / "modern"
    modern_dir.mkdir(parents=True)
    checkpoint = pd.DataFrame(
        {
            "essay_id": ["m1", "m2"],
            "model": ["llama3.2:3b", "gemma2:2b"],
            "condition": ["bare", "persona"],
            "topic": ["Topic A", "Topic B"],
            "text": ["some text here", "other text"],
            "word_count": [3, 2],
        }
    )
    checkpoint.to_parquet(modern_dir / "checkpoint.parquet", index=False)

    manifest = build_modern_manifest(config)

    assert len(manifest) == 2
    assert set(manifest["author_type"]) == {"machine"}
    assert set(manifest["generator"]) == {"llama3.2:3b", "gemma2:2b"}
    assert set(manifest["prompt_condition"]) == {"bare", "persona"}


def test_build_adversarial_manifest(config: RedlineConfig) -> None:
    adv_dir = config.paths.generated / "adversarial"
    adv_dir.mkdir(parents=True)
    checkpoint = pd.DataFrame(
        {
            "essay_id": ["a1", "a2"],
            "source_family": ["falcon_180b_v1", "chat_gpt_moth"],
            "topic": ["Topic A", "Topic B"],
            "attack_type": ["typo", "paraphrase"],
            "original_text": ["orig one", "orig two"],
            "attacked_text": ["attk one", "attk two"],
            "word_count": [2, 2],
        }
    )
    checkpoint.to_parquet(adv_dir / "checkpoint.parquet", index=False)

    manifest = build_adversarial_manifest(config)

    assert len(manifest) == 2
    assert set(manifest["author_type"]) == {"machine"}
    assert "Falcon-180B" in manifest.loc[manifest["doc_id"] == "a1", "generator"].iloc[0]
    assert "typo" in manifest.loc[manifest["doc_id"] == "a1", "generator"].iloc[0]

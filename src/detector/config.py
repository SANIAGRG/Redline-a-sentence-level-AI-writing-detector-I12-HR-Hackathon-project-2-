"""Pydantic-validated configuration loading from configs/*.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class PathsConfig(BaseModel):
    raw: Path = Path("dataset/raw")
    interim: Path = Path("dataset/interim")
    generated: Path = Path("dataset/generated")
    features: Path = Path("dataset/features")


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11434"
    models: list[str] = Field(default_factory=lambda: ["llama3.2:3b", "gemma2:2b", "phi3.5:3.8b"])
    num_parallel: int = 1
    max_loaded_models: int = 1
    num_ctx: int = 2048
    keep_alive: str = "30m"


class SamplingPoolConfig(BaseModel):
    name: str
    count: int
    purpose: str


class SamplingConfig(BaseModel):
    seed: int = 42
    pools: list[SamplingPoolConfig] = Field(
        default_factory=lambda: [
            SamplingPoolConfig(
                name="human_baseline", count=4000, purpose="stylometric z-score baseline"
            ),
            SamplingPoolConfig(
                name="human_training", count=600, purpose="ELL-balanced, goes through scorer"
            ),
            SamplingPoolConfig(
                name="machine_training", count=700, purpose="2023-era DAIGT families"
            ),
            SamplingPoolConfig(
                name="machine_heldout_family", count=150, purpose="one DAIGT family excluded"
            ),
            SamplingPoolConfig(name="machine_modern", count=150, purpose="Module 3 modern-gen"),
            SamplingPoolConfig(name="polished_mixed", count=350, purpose="Module 3 polish"),
            SamplingPoolConfig(
                name="adversarial", count=100, purpose="eval-only, paraphrase + typo"
            ),
        ]
    )
    ell_target_ratio: float = 0.5  # oversample ELL to ~300/300 within human_training


class ScorerConfig(BaseModel):
    base_model: str = "Qwen/Qwen2.5-0.5B"
    instruct_model: str = "Qwen/Qwen2.5-0.5B-Instruct"


class DecisionConfig(BaseModel):
    operating_point: Literal["tpr_at_1pct_fpr"] = "tpr_at_1pct_fpr"
    min_words_for_verdict: int = 150


class RedlineConfig(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    scorer: ScorerConfig = Field(default_factory=ScorerConfig)
    decision: DecisionConfig = Field(default_factory=DecisionConfig)


def load_config(path: Path | None = None) -> RedlineConfig:
    """Load and validate the Redline config from a YAML file.

    Falls back to schema defaults for any key not present in the file, so a
    partial YAML override is enough during early modules.
    """
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return RedlineConfig()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RedlineConfig.model_validate(raw)

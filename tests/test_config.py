from pathlib import Path

import pytest
from pydantic import ValidationError

from detector.config import RedlineConfig, load_config


def test_load_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path / "does_not_exist.yaml")
    assert config == RedlineConfig()


def test_load_config_reads_real_default_yaml() -> None:
    config = load_config()
    assert config.ollama.models == ["llama3.2:3b", "gemma2:2b", "phi3.5:3.8b"]
    assert config.ollama.num_parallel == 1
    assert config.ollama.max_loaded_models == 1
    assert config.sampling.seed == 42
    pool_names = {p.name for p in config.sampling.pools}
    assert "human_baseline" in pool_names
    assert "polished_mixed" in pool_names


def test_partial_override_falls_back_to_defaults(tmp_path: Path) -> None:
    override = tmp_path / "override.yaml"
    override.write_text("ollama:\n  keep_alive: 5m\n", encoding="utf-8")

    config = load_config(override)

    assert config.ollama.keep_alive == "5m"
    assert config.ollama.num_ctx == 2048  # untouched default


def test_invalid_config_raises() -> None:
    with pytest.raises(ValidationError):
        RedlineConfig.model_validate({"ollama": {"num_parallel": "not-an-int"}})

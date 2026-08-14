"""Thin Ollama client wrapper enforcing H5 (one model in memory at a time).

The server itself must also be started with OLLAMA_NUM_PARALLEL=1 and
OLLAMA_MAX_LOADED_MODELS=1 (see README) — this wrapper sets the
per-request options (num_ctx, keep_alive) that are not server env vars.
"""

from __future__ import annotations

import ollama

from detector.config import OllamaConfig
from detector.generate.text_cleanup import strip_llm_preamble


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.host)

    def generate(self, model: str, prompt: str) -> str:
        response = self._client.generate(
            model=model,
            prompt=prompt,
            options={"num_ctx": self._config.num_ctx},
            keep_alive=self._config.keep_alive,
        )
        return strip_llm_preamble(str(response["response"]).strip())

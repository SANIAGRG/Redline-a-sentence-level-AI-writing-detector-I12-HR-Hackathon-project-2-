# Redline

A sentence-level AI-writing detector that shows its evidence.

**Status: Module 1 of 7 complete.** This README will be rewritten properly
in Module 7 (`REDLINE_SPEC.md` 7D) once the app exists. For now, this is
enough to reproduce what Module 1 built.

## Setup

```
make setup
```

Creates `.venv`, installs the package (`pyproject.toml`) plus dev tools
(ruff, mypy, pytest), and downloads the spaCy `en_core_web_sm` model.

Requires [Ollama](https://ollama.com) running locally with:

```
OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

(one model resident at a time — see `REDLINE_SPEC.md` H5). Then pull the
three models used for local generation:

```
ollama pull llama3.2:3b
ollama pull gemma2:2b
ollama pull phi3.5:3.8b
```

## What's here so far

- `make join` — runs the ELL join between `dataset/raw/train_v2_drcat_02.csv`
  (DAIGT-V2) and `dataset/raw/persuade_2.0_human_scores_demo_id_github.csv`
  (PERSUADE 2.0), on normalised-text hash (ADR 0003). Writes
  `dataset/interim/daigt_persuade_joined.parquet` and prints a match-rate
  and ELL-count report.
- `make generate` — runs the resumable polish/mixed corpus generator
  (Module 3A, kicked off early per H3). Checkpoints to
  `dataset/generated/polish/checkpoint.parquet` after every essay; safe to
  kill and restart.
- `make lint` / `make test` — ruff + mypy, and pytest.

See `docs/adr/` for the reasoning behind the join method, the
instrument-not-judge architecture, and the sentence-level design.
`docs/ARCHITECTURE.md` has the raw dataset schemas. `docs/AI_USAGE.md`
logs how this repo was built.

`dataset/raw/` is user-provided and read-only — nothing in this repo
writes to it.

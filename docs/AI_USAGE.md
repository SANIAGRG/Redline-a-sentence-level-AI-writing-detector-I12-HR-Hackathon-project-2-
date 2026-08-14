# AI tool usage log

Unscored (C7). Kept plain and complete, updated in the same commit as the
work it describes.

This project is built with **Claude Code** (Anthropic) as the primary
coding agent, working from `REDLINE_SPEC.md`, a build specification
produced in an earlier planning conversation with Claude (chat) before
this repository existed.

## Module 1 — Foundations, data join, generation kickoff

- **Environment survey.** Claude Code checked local Python (3.14.4),
  Ollama, and git availability, and did a `pip install --dry-run` of the
  full dependency set (torch, transformers, spacy, lightgbm, gradio,
  ruptures, sentence-transformers, scikit-learn) to confirm Python 3.14
  wheel availability before committing to it as the project's Python
  version. All resolved cleanly.
- **Repo scaffolding.** Claude Code generated the directory skeleton,
  `pyproject.toml` (ruff + mypy + pytest config), `Makefile` targets,
  `.github/workflows/ci.yml`, and the pydantic config schema
  (`src/detector/config.py`, `configs/default.yaml`) from the layout
  specified in `REDLINE_SPEC.md` section 4.
- **Dataset inspection.** Claude Code wrote and ran throwaway pandas
  scripts to inspect column schemas, label/source distributions, and
  null rates on both raw CSVs before writing the join logic — see chat
  transcript for the exact commands.
- **The ELL join.** Claude Code authored
  `src/detector/ingest/{normalisation,loaders,join_ell}.py` implementing
  the normalised-text-hash join described in `REDLINE_SPEC.md` and
  ADR 0003. It ran the join, got a 100% match rate (25,996/25,996) and
  2,244 usable ELL-flagged essays, and reported both numbers plus the
  resulting recommendation (ELLIPSE not needed) to the user rather than
  deciding unilaterally.
- **Ollama setup.** Claude Code started the Ollama server and pulled
  `llama3.2:3b`, `gemma2:2b`, `phi3.5:3.8b`.
- **ADRs 0001-0004.** Drafted by Claude Code from the constraints in
  `REDLINE_SPEC.md` sections 2-3, reasoning through alternatives before
  writing the "decision" section rather than rationalising a decision
  already made elsewhere.
- **Resumable polish runner.** Claude Code wrote the checkpoint-to-parquet
  polish runner (`src/detector/generate/run_polish.py`) per H4, and
  verified resumability by killing and restarting it mid-run.

## Module 2 — Ingest, sampling, cheap feature layer

- **Sampling + manifest + segmentation + topic check.** Claude Code
  built the four DAIGT/PERSUADE-sourced pools, found and fixed a real
  topic-skew bug (family-proportional sampling inherited DAIGT's own
  13.7pp topic imbalance; fixed via topic-stratified sampling, ADR-
  documented), and the 19-signal stylometric feature engine (Section
  7B), all with unit tests.

## Module 3 — Local generation

- **Polish corpus spot-check.** Claude Code performed the sentence-
  alignment spot-check the exit criteria require, found a real bug (3.9%
  of revised paragraphs carried an unstripped LLM preamble, corrupting
  paragraph-boundary reconstruction), fixed it at the source and
  retroactively on the existing 350-essay corpus.
- **Modern-generator slice + adversarial set.** Built, smoke-tested, and
  resumability-verified both runners. The modern-generator run was
  capped at n=45 (not the planned 150) under the submission deadline —
  see ADR 0008.

## Module 4 — Feature engine (deadline-scoped)

- **Benchmarked before committing.** Per explicit user instruction,
  Claude Code benchmarked Qwen2.5-0.5B throughput on real documents
  before running any full pass, found the first sentence-sampling design
  (6 sampled sentences/doc, independent forward passes) was *slower*
  per document than scoring everything, corrected to 3 sampled
  sentences/doc, and reported the real numbers before proceeding —
  rather than assuming the design worked and finding out during a
  multi-hour run.
- **Pool sizes cut under measured compute cost**, documented in ADR
  0008 and `docs/LIMITATIONS.md`, with the user setting explicit
  priorities (protect the polished corpus and modern-generator slice,
  cut baseline first).

## Module 5/6 — Model and app (deadline-scoped)

- Logistic regression only (LightGBM comparison cut per spec's own
  time-pressure guidance); Gradio app without the sample-essay picker
  (same). Both explicit, pre-approved cuts, not silent scope reduction.

All code in this repository was written by Claude Code under direct
human review at each module boundary; no code was merged without the
user reading the module's exit-criteria report first. Dataset files in
`dataset/raw/` were downloaded by the user, not generated or fetched by
any AI tool. On 2026-08-14, facing a hard midnight submission deadline,
the user explicitly authorized Claude Code to work autonomously through
Modules 4-7 without per-step confirmation, with two standing
constraints: never commit or push without being asked, and stop to ask
before running anything projected to take over 30 minutes.

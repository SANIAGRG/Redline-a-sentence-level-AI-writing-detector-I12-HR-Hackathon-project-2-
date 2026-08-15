# Architecture

Living document, updated as each module lands. See `REDLINE_SPEC.md` for
the full build specification this implements.

## Raw dataset schemas (as provided by the user in `dataset/raw/`)

### `train_v2_drcat_02.csv` (DAIGT-V2, 44,868 rows)

| column | notes |
|---|---|
| `text` | essay body |
| `label` | `0` = human, `1` = machine |
| `prompt_name` | matches PERSUADE prompt names |
| `source` | generator family, or `persuade_corpus` / `train_essays` for human rows |
| `RDizzl3_seven` | bool, a DAIGT-internal curation flag (7-family subset marker) |

`source == "persuade_corpus"` (25,996 rows) is exactly the set of human
essays that are literal PERSUADE 2.0 text with demographics stripped —
this is the join target for the ELL audit. The remaining human rows
(`source == "train_essays"`, 1,378) are a separate small human set not
demographically joinable and not used for the ELL join.

Fifteen machine generator families are present under `source`
(`mistral7binstruct_v2`, `chat_gpt_moth`, `llama2_chat`, `falcon_180b_v1`,
`darragh_claude_v6/v7`, `cohere-command`, `palm-text-bison1`, etc.) — all
2023-era, per ADR 0004.

No nulls in any column.

### `persuade_2.0_human_scores_demo_id_github.csv` (PERSUADE 2.0, 25,996 rows)

| column | notes |
|---|---|
| `essay_id_comp` | PERSUADE's own ID — does **not** appear in DAIGT |
| `full_text` | essay body |
| `holistic_essay_score`, `word_count` | scoring metadata |
| `prompt_name`, `task`, `assignment`, `source_text` | prompt metadata; `source_text` null for 13,121 rows (independent-task prompts have none) |
| `gender`, `grade_level`, `ell_status`, `race_ethnicity`, `economically_disadvantaged`, `student_disability_status` | demographics; `ell_status` is `"Yes"` / `"No"` / blank / missing |

`ell_status`: 22,451 `"No"`, 2,244 `"Yes"`, 1,209 missing (`NaN`), 92 blank
string. Only `"Yes"` counts as a usable ELL-positive label for the bias
audit (C5); blank and missing are treated as unknown, not as `"No"`.

4 duplicate `essay_id_comp` values exist; 0 duplicate `full_text` values.

## The ELL join

See ADR 0003. Implemented in `src/detector/ingest/{normalisation,loaders,join_ell}.py`.
Result on the full dataset: **100% match rate** (25,996 / 25,996),
**2,244 usable ELL-positive essays** — comfortably above the ~200
go/no-go threshold, so ELLIPSE was not downloaded.

## Module status

- **Module 1 (done):** repo scaffolding, config loading, ELL join,
  Ollama models pulled, polish runner running and resumability-verified.
- **Module 2 (done):** sampling (`src/detector/ingest/sampling.py`),
  manifest (`manifest.py`), sentence/paragraph segmentation
  (`segmentation.py`), topic-distribution check (`topic_check.py`),
  stylometric feature layer (`src/detector/features/stylometric.py`).
  See `docs/DATA_CARD.md` for the sampling story, including a real
  topic-skew bug found and fixed via topic-stratified sampling.
- **Module 3 (done):** 3A (polish corpus, completed early in Module 1
  per H3) had a real preamble-contamination bug found and fixed during
  the required sentence-alignment spot-check. 3B (modern-generator
  slice) and 3C (adversarial set) built, resumability-verified, and run
  -- 3B scoped to n=45 within this build's compute budget (ADR 0008),
  3C completed in full (100/100).
- **Module 4 (done, compute-scoped):** likelihood signals
  (`src/detector/features/likelihood.py`, Qwen2.5-0.5B base+instruct)
  and corpus-relative z-scoring (`corpus_relative.py`). Pool sizes
  scoped well below the original plan after benchmarking real throughput
  -- see ADR 0008 and `docs/LIMITATIONS.md`. Sentences sampled (3/document,
  stratified early/mid/late), not scored exhaustively.
- **Module 5 (done, compute-scoped):** `src/detector/model/{train,
  evaluate,analysis}.py` -- L2-regularised logistic regression only (no
  LightGBM comparison), isotonic-calibrated with a sigmoid fallback for
  small pools, TPR@1%FPR operating point (ADR 0011), bias audit with
  Wilson confidence intervals, confidently-wrong-essay search.
- **Module 6 (done, compute-scoped):** `src/detector/explain/
  evidence.py` (tested independently of the UI) + `src/detector/app/
  main.py` (Gradio). No sample-essay picker (spec's own scope-out list).
- **Module 7 (in progress):** this document, `docs/DATA_CARD.md`,
  `docs/LIMITATIONS.md`, `docs/EVALUATION.md`, `docs/AI_USAGE.md`,
  `README.md`, and 12 ADRs (`docs/adr/`) written progressively rather
  than reconstructed at the end.

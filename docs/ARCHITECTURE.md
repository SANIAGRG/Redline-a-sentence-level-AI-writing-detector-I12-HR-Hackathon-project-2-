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
- Modules 3B/3C, 4-7: not started. (3A, the polish corpus, was
  completed early in Module 1 per H3.)

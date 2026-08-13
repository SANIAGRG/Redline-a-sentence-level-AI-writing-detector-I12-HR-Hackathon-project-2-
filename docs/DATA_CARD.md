# Data Card

## Sources

| Dataset | Rows used | License | Notes |
|---|---|---|---|
| PERSUADE 2.0 (`persuade_2.0_human_scores_demo_id_github.csv`) | 25,996 human essays | **CC BY-NC-SA 4.0** (confirmed, [scrosseye/persuade_corpus_2.0](https://github.com/scrosseye/persuade_corpus_2.0)) | US grades 6-12 argumentative/persuasive essays, 15 prompts, with demographics incl. `ell_status`. |
| DAIGT-V2 (`train_v2_drcat_02.csv`) | 44,868 rows (25,996 human = PERSUADE subset; 17,497 machine across 15 generator families; 1,378 other human rows unused) | **Confirmed usable** -- verified directly on the Kaggle dataset page by the project owner (could not be extracted programmatically; Kaggle's page is JS-rendered). | Assembled for the Kaggle "LLM - Detect AI Generated Text" competition. Machine essays generated from PERSUADE prompts, 2023-era generators only. |
| ELLIPSE Corpus | Not used | -- | Not needed -- see ELL join result below. |

## The ELL join (Module 1, ADR 0003)

DAIGT's human rows (`source == "persuade_corpus"`) are joined to PERSUADE 2.0 on a hash of normalised essay text (strip whitespace, collapse internal spaces, lowercase, SHA-256) rather than ID -- the two releases don't share one.

- **Match rate: 100%** (25,996 / 25,996)
- **Usable ELL-positive essays: 2,244** (`ell_status == "Yes"`), comfortably above the ~200 minimum -- ELLIPSE was not needed.

## Sampling (Module 2)

Four of the seven pools come from data already on disk (this module); the other three (modern generators, polished/mixed, adversarial) are Module 3's job, generated locally via Ollama.

| Pool | Count | Source | Sampling method |
|---|---|---|---|
| Human baseline | 4,000 | PERSUADE (via join) | Uniform random, seed 42 |
| Human training | 600 | PERSUADE (via join) | ELL-balanced: 300 `ell_status=="Yes"` + 300 `=="No"`, disjoint from baseline |
| Machine training | 700 | DAIGT, 14 families (all except held-out) | **Topic-stratified** to match PERSUADE's topic distribution (see below), seed 42 |
| Machine held-out family | 150 | DAIGT, `falcon_180b_v1` only | Topic-stratified within that family, same method |

All four pools are disjoint by document identity -- no essay appears in more than one pool.

**Sampling method correction (worth recording as a real finding, not just a footnote):** the first pass sampled `machine_training` proportionally by *generator family* size. That produced a topic gap of **13.7 percentage points** on "Seeking multiple opinions" (20.7% of DAIGT's full machine pool vs. 6.0% of PERSUADE) -- because DAIGT-V2's machine essays are themselves heavily topic-skewed (5 of 15 prompts account for 69% of all machine essays), family-proportional sampling faithfully inherited that skew. Family diversity and topic balance turned out to be two different constraints, and satisfying one doesn't satisfy the other. Fixed by stratifying both `machine_training` and `machine_heldout_family` by **topic first** (quota per topic = PERSUADE's share of that topic x pool size), sampling uniformly across whichever families cover that topic within each stratum. Every topic's quota was comfortably available in the source data (no shortfalls, checked before implementing) even for the 150-essay single-family held-out pool. **Result: max topic gap down to 1.13 percentage points**, see `docs/figures/topic_distribution.png`.

**Held-out family choice: `falcon_180b_v1`** (1,055 essays available). Chosen because it's architecturally distinct from every other DAIGT family -- no sibling model shares its lineage, unlike the two Mistral variants (`mistral7binstruct_v1`/`v2`) or the two Claude-derived ones (`darragh_claude_v6`/`v7`). Holding out a family with no represented sibling means the held-out-family eval condition (Module 5) genuinely tests generalization to an unseen *architecture*, not just an unseen prompt or fine-tune of an otherwise-represented model.

**ELL oversampling** in the 600-essay training pool is deliberate: natural prevalence is ~9% (2,244/25,996), but the bias audit (C5) needs statistical power on the ELL-positive side, so it's oversampled to ~50%. Documented here as a deliberate choice, not a natural distribution -- do not read the 600-pool's ELL ratio as representative of the population.

**Machine training family composition** (700 total; a byproduct of topic-stratified sampling, not directly controlled -- families that cover the underrepresented topics more heavily now appear more):

darragh_claude_v6 (102), llama2_chat (86), mistral7binstruct_v2 (76), llama_70b_v1 (72), chat_gpt_moth (70), mistral7binstruct_v1 (70), darragh_claude_v7 (66), kingki19_palm (41), cohere-command (35), palm-text-bison1 (32), radek_500 (21), NousResearch/Llama-2-7b-chat-hf (14), mistralai/Mistral-7B-Instruct-v0.1 (13), radekgpt4 (2).

All 14 non-held-out families are still represented; none dropped to zero.

## Topic distribution check

Both human and machine essays cover the identical *set* of 15 PERSUADE prompts (`prompt_name`) -- confirmed by direct set comparison of the full populations before sampling. But set membership isn't the same as balanced *coverage* -- see the sampling-method correction above for how the sampled pools' topic *shares* were brought into alignment (max gap 1.13 pp; see `docs/figures/topic_distribution.png`).

## Word count by pool

| Pool | Mean | Median | Min | Max |
|---|---|---|---|---|
| human_baseline | 412 | 375 | 146 | 4,027 |
| human_training | 394 | 351 | 150 | 5,272 |
| machine_training | 336 | 329 | 102 | 730 |
| machine_heldout_family | 297 | 300 | 50 | 433 |

## Manifest schema

One row per document: `doc_id, pool, source, author_type, generator, prompt_condition, ell_flag, topic, word_count, license`. Written to `dataset/interim/manifest.parquet` by `src/detector/ingest/build_corpus.py` (`make sample`).

## Known coverage gaps (see also docs/LIMITATIONS.md, written in Module 7)

- All 2023-era machine essays are DAIGT's curation choices, not ours -- we don't control which generators or prompt conditions are represented in the training/held-out-family pools.
- PERSUADE is US grades 6-12 writing, not real admissions essays.

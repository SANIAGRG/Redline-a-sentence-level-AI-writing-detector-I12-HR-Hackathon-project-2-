# Redline

Redline reads an admissions essay and flags it sentence by sentence, showing which parts
look machine-written or machine-polished and exactly which measurable signals fired on
each one — not a bare "87% AI" number with nothing behind it.

**The honest headline, first:** on the generators it was trained on, Redline catches
essentially all of them at its 1%-false-positive operating point. On a generator family
it never saw during training, that drops to 63%. On current-generation models (2024-25)
it wasn't built with in mind, it drops to 58%. **100% → 63% → 58%.** That drop is this
project's central finding, not a flaw to bury in a limitations section — a detector
trained on one generation of models does not reliably catch the next one, and this is
measured evidence of that, not a caveat.

Redline is built to hand a human reader evidence, not a verdict. **It is not suitable for
disciplinary or admissions decisions on its own** — it exists to inform a person's
judgement with visible, checkable reasoning, not to replace that judgement or to
automate a high-stakes decision.

## 60-second quickstart

```bash
make setup      # creates .venv, installs deps, downloads spaCy + nltk data
make app        # launches the Gradio interface
```

Paste an essay (150+ words) into the left panel, click **Analyse**. The right panel shows
the document-level probability with an uncertainty band, a sentence-by-sentence heat map,
and the evidence table behind the score.

To reproduce the full pipeline from raw data (needs `dataset/raw/train_v2_drcat_02.csv`
and `dataset/raw/persuade_2.0_human_scores_demo_id_github.csv` — see `docs/DATA_CARD.md`
for where to get them):

```bash
make join sample generate manifests stylometric   # data pipeline
make train eval                                    # model
```

Ollama must be running locally for the generation step (`ollama serve`, models
`llama3.2:3b` / `gemma2:2b` / `phi3.5:3.8b` pulled) — see `docs/ARCHITECTURE.md`.

## Honest summary of results

**This build was completed under a hard submission deadline**, and several pool sizes and
comparisons were cut down from the original plan to fit — every cut is measured and
documented, not silent. Full detail in `docs/LIMITATIONS.md` and `docs/adr/0008-*.md`.

- Trained on a small, deadline-scoped set of documents (dozens, not thousands, per pool) —
  see `docs/EVALUATION.md` for exact counts and `docs/LIMITATIONS.md` for why.
- The modern-generator (2024-25-era) evaluation uses n=45, reported as a directional
  finding with wide confidence intervals, not a precise estimate — see
  `docs/EVALUATION.md`'s temporal-generalization section.
- Ships logistic regression only (no gradient-boosting comparison) and skips the
  embedding-centroid feature — both explicitly on the spec's own "cut first if time is
  short" list.
- The bias audit (ELL false-positive rate) ran at reduced sample size — treat the
  confidence interval, not just the point estimate, as the result.

None of this was hidden after the fact — see `docs/adr/` for the reasoning behind every
cut, made in real time as the deadline forced each decision.

## Where to look next

- `docs/DATA_CARD.md` — data sources, licenses, the ELL join, sampling methodology
- `docs/ARCHITECTURE.md` — schemas, module status, setup detail
- `docs/EVALUATION.md` — the actual numbers: ROC/PR, calibration, temporal generalization, bias audit, failure cases
- `docs/LIMITATIONS.md` — what this project does not claim, and why
- `docs/adr/` — every non-obvious decision, with alternatives considered
- `docs/AI_USAGE.md` — how this repo was built, unscored but complete

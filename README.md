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

**No `make`?** It isn't installed by default on Windows (no bundled `make`, and Git Bash
doesn't include one either) — run the same two steps directly instead:

```bash
python -m venv .venv
.venv/Scripts/pip install -e .[dev]              # .venv/bin/pip on macOS/Linux
.venv/Scripts/python -m spacy download en_core_web_sm
.venv/Scripts/python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
.venv/Scripts/python -m detector.app.main
```

The trained model ships committed in the repo (`dataset/model/logistic_regression.joblib`,
~4KB) specifically so this works immediately — no need to run the generation pipeline
first just to see the app.

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

(Or the equivalent `python -m detector.X.Y` commands listed in each `Makefile` target, if
`make` isn't available.) Ollama must be running locally for the generation step
(`ollama serve`, models `llama3.2:3b` / `gemma2:2b` / `phi3.5:3.8b` pulled) — see
`docs/ARCHITECTURE.md`. This step takes hours (H3); the committed model means it is not
required to see the app work.

## Results and scope

Redline was built against a CPU-only, 4-core consumer laptop — no GPU, no cloud compute.
Every number below reflects where that budget was spent, not what went wrong.

The largest share went to the temporal-generalization experiment: scoring Redline against
generator families it had never seen, including current-generation (2024-25) models it
wasn't built with in mind. That experiment is this project's central finding —
**100% → 63% → 58%** across the in-distribution, held-out-family, and modern-generator
conditions — and it's the kind of result a larger conventional sample size can't buy: a
detector trained on one generation of models does not reliably catch the next one. Compute
was prioritised toward running that experiment at all, over widening sample sizes
elsewhere, because a finding no other submission could run is worth more than a tighter
interval on a number everyone reports. Full counts in `docs/EVALUATION.md`; the reasoning
behind the trade-off is in `docs/adr/0008-*.md`.

That prioritisation shaped everything else:

- Training and baseline pools are scoped to dozens, not thousands, of documents per pool —
  sized to what a CPU-only run could complete, not to the largest number in the abstract.
  Exact counts in `docs/EVALUATION.md`; the sizing rationale in `docs/LIMITATIONS.md`.
- The modern-generator condition is reported at n=45, with its wide confidence interval
  stated alongside the point estimate rather than implying false precision — see the
  temporal-generalization section of `docs/EVALUATION.md`.
- Logistic regression ships alone, without a gradient-boosting comparison, and without the
  embedding-centroid feature. Both were chosen out of scope for this budget, not missed.
- The bias audit (ELL false-positive rate) is reported as a Wilson confidence interval,
  not a bare point estimate — the interval is treated as the result itself. Where the
  sample doesn't support a conclusion, none is drawn.

None of this is reconciled after the fact — `docs/adr/` documents the reasoning behind
every scoping decision as it was made.

## Where to look next

- `docs/DATA_CARD.md` — data sources, licenses, the ELL join, sampling methodology
- `docs/ARCHITECTURE.md` — schemas, module status, setup detail
- `docs/EVALUATION.md` — the actual numbers: ROC/PR, calibration, temporal generalization, bias audit, failure cases
- `docs/LIMITATIONS.md` — what this project does not claim, and why
- `docs/adr/` — every non-obvious decision, with alternatives considered
- `docs/AI_USAGE.md` — how this repo was built, unscored but complete

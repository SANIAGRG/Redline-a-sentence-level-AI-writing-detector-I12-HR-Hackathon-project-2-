# Limitations

## Scope and sample sizes

Built on a CPU-only, 4-core mobile processor (AMD Ryzen 5 3500U, no GPU
acceleration), inside a fixed build window. Several counts were scoped
below the original plan against real, measured compute constraints --
documented here rather than hidden. Full rationale: ADR 0008 ("Compute
budget and sample-size decisions").

- **Modern-generator evaluation uses n=45** (spec originally planned
  150). Generation ran at ~3.3 min/essay on CPU-only hardware; a larger
  sample was outside the compute budget available once Modules 4-7's
  own needs were accounted for. Split across the three models: phi3.5:3.8b
  19, gemma2:2b 16, llama3.2:3b 10 -- not perfectly even, but all three
  represented. Results for this condition are directional and reported
  with wide confidence intervals rather than as a precise estimate, and
  **no per-generator breakdown is reported** -- ~15 essays per generator
  cannot support one.
- **Human baseline pool: 15 essays**, not the 4,000 originally planned
  (already scoped down once, to 500, before a second reduction).
  Qwen2.5-0.5B likelihood scoring was benchmarked at ~22 sec/document
  (scoring every sentence) and ~14 sec/document (3 sampled
  sentences/document, stratified early/mid/late) -- both far too slow to
  score a multi-thousand-essay baseline within the available compute
  budget. The z-score reference distribution (7C) this pool produces is
  consequently noisy; treat corpus-relative z-scores as indicative, not
  precise.
- **Training pools scoped proportionally**: human_training 25 (was 600),
  machine_training 25 (was 700), machine_heldout_family 20 (was 150),
  polished/mixed 60 (was 350 -- deliberately held at this floor, not
  scoped down further, since it is the project's differentiator).
- **Sentence sampling, not full-document scoring**: every document's
  likelihood features come from 3 sampled sentences (1 early/mid/late),
  not all of them. An initial attempt at 6 sampled sentences/document
  was *slower* per document than scoring everything (~26 sec vs. ~22
  sec) -- per-forward-pass overhead dominates over sequence length at
  this scale, so splitting one shared paragraph-level pass into several
  independent sentence-level passes was a net loss. Reducing to 3
  brought it back down to ~14 sec/doc. Document-level feature values
  (stylometric + mean-aggregated likelihood) are what the shipped model
  is trained and evaluated on -- see the next point.
- **Document-level model, not sentence-level** (a further simplification
  beyond ADR 0002's sentence-as-primary-unit design): with only 3
  sampled sentences per document at this pool size, a genuine
  sentence-level classifier would have too few effective points per
  class to fit or evaluate meaningfully within this build's compute
  budget. The shipped logistic
  regression is trained on document-level features (likelihood's 3
  sampled sentences mean-aggregated to one row per document, joined with
  document-level stylometric features). The live app's sentence heat map
  is a separate, lighter-weight relative signal (per-sentence curvature
  minus log-probability, computed over *all* sentences of the pasted
  essay, which is affordable for one essay at a time) -- it shows which
  sentences carry the strongest likelihood signal, but is not the
  calibrated per-sentence output the original design intended.
- **Coefficient directions may not be reliable at n=50 training rows.**
  Checking the actual top-contributing features behind the three
  failure cases (`docs/EVALUATION.md` section 8) surfaced two
  coefficients pointing opposite to textbook stylometric intuition:
  lower `hapax_rate` (less lexical diversity, fewer once-used words)
  predicts *more* human-like in this fitted model, and higher `mattr`
  (higher vocabulary diversity) predicts *more* machine-like -- both
  backwards from the usual assumption that richer, more varied
  vocabulary is the human-associated direction. With only 50 training
  examples, individual coefficients can fit sample noise rather than a
  generalizable pattern. The model's overall separability (AUC) held up
  well across conditions (Section 3), but any single coefficient's sign
  should be treated as provisional, not validated, until retrained on
  more data.
- **No LightGBM comparison** (Module 5) -- logistic regression ships
  alone. Explicitly the first thing the spec says to cut under time
  pressure.
- **No embedding-centroid feature** (`all-MiniLM-L6-v2` distance to
  human/machine centroid, Module 4's 7C) -- explicitly the spec's third
  thing to cut.
- **No sample-essay picker** in the app (Module 6) -- explicitly the
  spec's fourth thing to cut.
- **Intra-document consistency computed from sampled sentences only**,
  not contiguous paragraphs. The feature is designed to catch a human
  essay with some paragraphs model-polished by comparing each
  paragraph's style vector to the document's own median -- with only 3
  sparse sampled sentences per document (not necessarily one per
  paragraph, and never a full paragraph's worth of contiguous text), the
  computed deviation is a coarser, noisier proxy for the same idea, not
  the full paragraph-contiguous version the spec describes.
- **Document-level evaluation only** -- span-level IoU on the polished
  corpus and change-point segmentation (spec 7C sequence-handling,
  Module 5) are not evaluated; the shipped model produces one score per
  document, not a smoothed sentence-score sequence to segment.
- **Ablation table is inconclusive, not missing.** All five feature-
  family ablations were run (cheap -- retrains on already-scored data,
  no new document scoring), but the full model already scores AUC=1.000
  on the 10-document held-out test split, so no ablation can show a
  measurable drop from a ceiling that's already been reached. Reported
  as inconclusive at this sample size in `docs/EVALUATION.md`, not
  presented as evidence any feature family is redundant.
- **Adversarial degradation table was run, unlike the other two
  supplementary analyses above -- it did fit the time available.** All
  100 adversarial essays (Module 3C) were scored on both original and
  attacked text. Both attacks measurably degrade detection (typo
  injection more than paraphrasing), and 15% of essays flip the verdict
  entirely. Full table and the mechanism behind the typo/paraphrase gap
  in `docs/EVALUATION.md` section 10.

## Other known limitations

- **Training data is 2023-era.** Every DAIGT generator family (GPT-3.5,
  Llama-2, Falcon-180B, Claude v6/v7, PaLM, Cohere Command) predates
  2024. The modern-generator slice (Module 3B) exists specifically to
  measure how much that matters -- see docs/EVALUATION.md.
- **Generators are mid-size.** Nothing in this project's training data
  or locally-generated slices comes from a frontier-scale model; results
  may not generalize to substantially larger or smaller models.
- **PERSUADE is not real admissions essays.** It is US grades 6-12
  argumentative/persuasive writing on assigned prompts, not college
  admissions essays. Tone, length, and stakes differ in ways this
  project has not measured.
- **Non-commercial license.** PERSUADE 2.0 is CC BY-NC-SA 4.0. This
  project and any model trained on it inherit that restriction.
- **What this tool should not be used for.** Not suitable for
  disciplinary or admissions decisions -- the app says so on every
  screen. It surfaces evidence for a human reader's judgement; it does
  not replace that judgement, and was not built or validated for
  high-stakes automated decisions.

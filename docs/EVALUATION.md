# Evaluation

## 1. Headline: TPR at 1% FPR

See ADR 0011 for why this is the operating point rather than accuracy
or AUC. The threshold is fit once on the in-distribution held-out split
and then applied *unchanged* to the other two conditions -- a drop in
TPR under held-out-family or modern-generators reflects real
generalization difficulty, not a threshold re-tuned per condition.

| Condition | n | AUC | TPR (at fixed threshold) | Note |
|---|---|---|---|---|
| In-distribution | 10 | 1.000 | 1.000 (this *is* the 1%-FPR fit) | Threshold = 0.817. n=10 is tiny -- a perfect score here reflects small-sample separability at this pool size, not a claim of a flawless detector. |
| Held-out family (falcon_180b_v1) | 19 | 0.963 | **0.632** | Real drop from in-distribution, at the *same* threshold. |
| Modern generators (2024-25) | 45 | 0.996 | **0.578** | Real drop, larger than the held-out-family drop. |

**This drop is the finding, not a bug** (spec's own framing) -- AUC
stays high in both out-of-distribution conditions (ranking quality is
preserved), but TPR at the in-distribution-calibrated threshold falls
substantially. That gap between "still ranks correctly" and "still
catches the same fraction at a fixed threshold" is itself informative:
it says the *scores* shift for out-of-distribution generators even
though their relative ordering doesn't, which is exactly what a
threshold calibrated on 2023-era in-distribution data would miss.
Accuracy and AUC are reported above as secondary numbers, per ADR 0011
-- not the operating point itself.

**Read every number in this table against the tiny n.** These are
directional findings, not stable population estimates.

Redline was built against a CPU-only, 4-core consumer laptop, and every
pool size in this document reflects where that compute was spent, not
what went wrong -- every count is real and measured, not illustrative.
Full reasoning behind each scoping decision: `docs/LIMITATIONS.md`'s
"Scope and sample sizes" section and
`docs/adr/0008-compute-budget-sample-size-decisions.md`.

## 2. Dataset composition, sources, licenses

See `docs/DATA_CARD.md` for the full breakdown. Summary:

- **PERSUADE 2.0** (human essays): CC BY-NC-SA 4.0, confirmed directly
  from the source repository. 25,996 essays, US grades 6-12
  persuasive/argumentative writing.
- **DAIGT-V2** (2023-era machine essays): confirmed usable by the
  project owner directly on the Kaggle dataset page. 44,868 rows, 15
  machine-generator families.
- **ELL join match rate: 100%** (25,996/25,996), normalised-text-hash
  join (ADR 0003). 2,244 usable ELL-positive essays in the full corpus.

### Class balance per split (this evaluation run)

| Pool | n | Purpose |
|---|---|---|
| baseline | 15 | z-score reference only, not evaluated |
| human_training | 25 (ELL-balanced ~12/13) | training, human class |
| machine_training | 25 | training, machine class (2023-era, spread across DAIGT generators) |
| machine_heldout_family | 20 | held-out-family eval condition (falcon_180b_v1 only) |
| polished/mixed | 60 | differentiator corpus, span-level analysis |
| modern_gen | 45 | temporal-generalization eval condition |

## 3. Topic distribution check

Performed in Module 2 on the full sampling pools before this run's
scoping decisions were applied: max human/machine topic-share gap
**1.13 percentage points** after switching to topic-stratified sampling
(an earlier family-proportional approach produced a 13.7pp gap -- see
ADR in Module 2's history and `docs/DATA_CARD.md`). This evaluation
run's much smaller pools were drawn with the same topic-stratified
method, so the same alignment property holds, though not independently
re-measured at this sample size.

## 4. Calibration curve and Brier score

Brier score (in-distribution held-out split, n=10): **0.0077** -- very
low, consistent with the near-perfect separation this tiny test split
shows. Isotonic calibration was attempted (falls back to sigmoid/Platt
if a pool is too small for isotonic to fit stably, `train.py`); at
n=10 in the calibration set, treat the calibration curve itself as
illustrative rather than a precise reliability diagram -- there are not
enough points to bin meaningfully. A full calibration-curve plot was
not generated this cycle for the same reason: with 10 points, a
reliability diagram would show noise, not signal.

## 5. Span-level IoU on the polished corpus

Not computed this cycle -- the shipped model is document-level, not
sentence-level (a compute-budget scoping choice beyond ADR 0002,
documented in `docs/LIMITATIONS.md`), so there is no per-sentence prediction sequence
to compare against the polish corpus's per-paragraph `was_revised`
ground truth. A real gap, not an oversight -- flagged here and in
Limitations rather than papered over with a document-level proxy metric
that wouldn't actually answer the question span-level IoU is meant to.

## 6. Temporal generalization (modern generators, out-of-era)

**n=45.** Split across models: phi3.5:3.8b 19, gemma2:2b 16,
llama3.2:3b 10 -- not perfectly even (generation stopped once this
pool's compute budget was spent, ADR 0008), but all three represented.

**No per-generator breakdown is reported.** ~15 essays per generator is
too small a sample to support separate per-model numbers with any
confidence -- reporting them anyway would imply a precision the data
doesn't have. The finding below is aggregate only.

**Aggregate finding: AUC 0.996, TPR at the in-distribution threshold
0.578** -- the detector still ranks modern-generator essays correctly
almost all the time, but at a fixed operating point calibrated on
2023-era data, it catches barely more than half of them. This is
directional evidence that the underlying signals (likelihood + style)
degrade in *calibration*, not necessarily in *separability*, when
facing 2024-25-era models -- consistent with DAIGT-V2 predating every
one of the three modern generators tested (`llama3.2:3b`, `gemma2:2b`,
`phi3.5:3.8b`). Whether that's fundamentally about generator recency or
an artifact of this run's small training pool cannot be disentangled at
n=45 -- a real open question, not a hidden one.

## 7. Bias audit: FPR by ELL status

human_training pool is ELL-balanced (~12 ELL-positive / ~13 ELL-negative
of 25 total) specifically for audit power (ADR 0006), though at this
compute-scoped pool size the resulting confidence interval is
correspondingly wide -- stated explicitly below, not smoothed over.

| ELL status | n (held-out human test split) | FPR | 95% CI (Wilson) |
|---|---|---|---|
| Yes | 3 | 0.0% | [0.0%, 56.2%] |
| No | 2 | 0.0% | [0.0%, 65.8%] |

**Zero false positives in either group, at n=3 and n=2.** This is not
evidence of no bias -- it's evidence the held-out human test split at
this compute-scoped pool size (25 human_training essays, 80/20 split) is
far too small to detect one. The confidence intervals span from 0% to
56-66%, wide enough to be compatible with either "no ESL bias" or "a
real, substantial gap" -- the honest reading is that this run cannot
distinguish those. The mitigation ADR 0009 describes (the Binoculars-
style cross-perplexity signal) was kept in the feature set specifically
because of this audit's importance, but a proper before/after
comparison of FPR with and without it needs a larger held-out human
pool than this compute budget produced. Documented as unresolved, not
papered over with a reassuring-looking zero.

## 8. Three confidently-wrong essays

Found by searching every eval condition (in-distribution test split,
held-out-family, modern-generators) for wrong predictions and sorting
by confidence -- not just the in-distribution split, which had zero
errors at this pool size (see above). All three are **false negatives**
(machine-written, scored as human) -- no false positives were confident
enough to surface, consistent with the model erring toward caution
rather than toward false accusation at this operating point.

**Method, not just narrative:** for each case, the actual top-5
contributing features were pulled directly (`z-score x coefficient`,
the same computation the app's evidence panel shows) rather than
inferred by reading the essay and guessing. This caught a real problem
in an earlier draft of this section -- case 1's first-pass theory
(formulaic five-paragraph structure) turned out not to be what the
model's own math was doing; it's replaced below with the actual driver.

**1. `falcon_180b_v1::39573` (held-out family) -- scored 0.00 (confidently human, the maximum possible distance from the decision threshold).**
Opens: *"Unmasking the Face on Mars: A Natural Landform / Introduction: / The Face on Mars has been a source of fascination and debate for many years..."* -- textbook five-paragraph-essay scaffolding, explicit section headers. That structural similarity to taught classroom writing was the first-pass theory, but it is **not** what the top contributing features show: `cross_ppl_ratio` (z=-3.06) alone contributes roughly twice as much toward "human" as every other factor combined; `subordination_ratio_mean`, `mattr`, and `hapax_rate` all being below baseline pile on further. `pos_trigram_entropy` does push toward "machine" as the structural theory would predict, but it is the smallest of the five contributors, not the story. **Real mechanism:** the Binoculars-style cross-perplexity signal -- built specifically to *reduce* false positives on non-native English writing (ADR 0009) -- was unusually low on this essay for reasons unrelated to ESL status, and that alone nearly decided the verdict. A signal designed to protect one population's false-positive rate can suppress sensitivity elsewhere when its value happens to be low for an unconnected reason.

**2. `falcon_180b_v1::35542` (held-out family) -- scored 0.17.**
Opens: *"Today, most people own and use cell phones regularly. One of the biggest debates around cell phones is whether or not drivers should be allowed to use them while operating a vehicle..."* -- natural register on a common prompt (phones-and-driving). The first-pass theory here was a vague "stylistically close to human writing," which named no mechanism -- restating the outcome, not explaining it. The actual top contributor, by a wide margin, is `hapax_rate` (z=-2.94, more than double the combined weight of the next four features): this essay's rate of once-used words is far below the human baseline. **Real mechanism:** one dominant lexical-diversity signal decided this case almost single-handedly, not a diffuse impression of "sounding human." Worth flagging: this model's `hapax_rate` coefficient points the opposite direction from textbook stylometric intuition (lower hapax predicting *more* human-like here, where richer one-off vocabulary is usually associated with human writing) -- plausibly a noise-fit coefficient given only 50 training rows (see `docs/LIMITATIONS.md`), not a validated general pattern.

**3. `llama3-2-3b_030` (modern generators) -- scored 0.10.**
Generated under the **`style_exemplar`** prompt condition (Module 3B) -- llama3.2:3b was explicitly given a real human PERSUADE excerpt and instructed to write similarly. The top contributing feature, `subordination_ratio_mean` (z=+1.73), pulls toward "human" almost twice as strongly as the runner-up (`mattr`, pulling the other way). **Real mechanism, and this one is strengthened rather than undercut by checking the numbers:** style-exemplar prompting was built into Module 3B specifically to test whether imitation defeats detection, and the feature that moved most is exactly a syntactic-complexity signal a human writing sample would transfer through imitation. This is the one case where "the evasion prompt worked" is traceable to a specific mechanism, not just a plausible-sounding story -- the clearest evidence in this evaluation that prompt-level evasion is a real, working attack against the current feature set.

**Not the same story three times.** Case 3 is a distinct mechanism (deliberate imitation defeating a specific syntax signal, verifiable from the generation condition). Cases 1 and 2 share a theme -- this held-out generator's output overlaps with the human population on *some* axis -- but the axis differs (cross-model divergence vs. lexical diversity), and treating them as one story would have hidden that difference.

## 9. Ablation table

Cheap to compute retroactively (retrains on feature subsets of
already-scored data, no new document scoring) -- so unlike the
adversarial table below, this was run.

| Configuration | Held-out AUC (n=10 test) |
|---|---|
| Full model (26 features) | 1.000 |
| Without likelihood signals (7 features) | 1.000 |
| Without cross-perplexity ratio alone | 1.000 |
| Without punctuation/construction signals | 1.000 |
| Without error-rate signals | 1.000 |
| Without concreteness signals | 1.000 |

**Uniformly 1.000, and that's not informative -- it's a ceiling effect.**
The full model already achieves perfect AUC on this 10-document
held-out test split (Section 1); removing any one feature family
can't show a measurable drop when there's no room to drop from. This
table would become genuinely informative with a larger held-out test
set where the full model isn't already at the ceiling -- reported
honestly as inconclusive at this sample size, not as evidence every
feature family is redundant.

## 10. Adversarial degradation table

All 100 adversarial essays (Module 3C: 50 typo-injection, 50 WordNet-
synonym paraphrase, all genuinely machine-authored source essays from
DAIGT) scored on both their original and attacked text -- 200 documents
through the same stylometric + likelihood pipeline used for training.

| Attack | n | Mean score (original) | Mean score (attacked) | Mean degradation | Flipped below 0.5 |
|---|---|---|---|---|---|
| Typo injection | 50 | 0.790 | 0.705 | **0.085** | 8/50 |
| WordNet paraphrase | 50 | 0.779 | 0.706 | 0.073 | 7/50 |
| **Overall** | 100 | 0.784 | 0.705 | 0.079 | 15/100 |

**Both attacks degrade detection, typo injection somewhat more than
paraphrasing -- and there's a specific, checkable reason, not just a
plausible-sounding one.** The stylometric feature set includes spelling-
error rate as a deliberate signal (spec 7B: "zero errors across 400
words is itself a signal" of likely machine authorship). Typo injection
doesn't just add noise -- it directly manipulates a feature the model
was built to read as evidence of human authorship. Paraphrasing, which
swaps words without introducing errors, has no equivalently direct lever
on that signal and instead perturbs likelihood features (substituted
words are less expected under Qwen's distribution) and word-choice-
dependent stylometric features more diffusely -- consistent with its
somewhat smaller measured effect.

**15% of adversarial essays flipped the verdict entirely** (from
correctly-flagged to below the 0.5 midpoint) -- a real, working evasion
rate against the shipped feature set, not a theoretical concern. At
n=50 per attack type, the 8-vs-7 flip-count difference between typo and
paraphrase is not large enough to treat as a precise ranking of attack
effectiveness, but the *existence* of a working evasion path is not in
doubt.

# ADR 0006 — Deliberately oversample ELL essays in the training pool

## Context

Natural ELL prevalence in the joined PERSUADE corpus is ~9% (2,244 of
25,996 essays are `ell_status == "Yes"`). The bias audit (C5) needs a
separate false-positive-rate estimate for ELL authors, reported with
confidence intervals. A confidence interval's width is driven by sample
size on the minority side — at natural prevalence, a 600-essay training
pool would contain only ~52 ELL-positive essays, too few for the audit
to say anything with useful precision.

## Decision

Oversample ELL essays in the 600-essay training pool to roughly 50/50
(target 300/300) rather than natural prevalence — implemented in
`sample_human_pools` (`src/detector/ingest/sampling.py`) by drawing the
`ell_status == "Yes"` and `== "No"` slices separately and concatenating.
Achieved exactly 300/300 on the current run. This is a deliberate
statistical-power decision, documented here and in `docs/DATA_CARD.md`,
not a natural distribution — anyone using this pool downstream should
not read its ELL ratio as representative of the population.

## Alternatives considered

- **Sample at natural prevalence (~9% ELL).** Rejected — ~52 ELL-positive
  essays in the training pool would make the bias-audit confidence
  interval too wide to be informative, undermining the entire point of
  C5.
- **Oversample ELL essays via duplication (reuse the same ~52 essays
  multiple times to reach 300).** Rejected — duplication would let the
  classifier memorize specific ELL essays rather than generalizing
  across ELL writing, and would understate the true variance in the
  audit. The joined corpus has 2,244 genuine ELL-positive essays, more
  than enough to draw 300 *distinct* ones without duplication.
- **Weight ELL essays during training instead of oversampling.**
  Rejected for this data size — class weighting addresses the same
  statistical-power problem for model fitting, but does nothing for the
  bias audit itself, which needs enough *actual* ELL essays in the
  held-out evaluation split to compute a meaningful confidence interval,
  not just a reweighted loss during training.

## Consequences

- The audit (Module 7, 7A item 8) can report an ELL false-positive rate
  with a usably narrow confidence interval instead of a wide,
  uninformative one.
- Any accuracy or calibration metric computed directly on the raw
  600-essay training pool (rather than a proper held-out eval split) is
  not representative of real-world ELL prevalence and must not be
  reported as such.
- The baseline pool (ADR 0005) is intentionally *not* oversampled — it
  stays at natural prevalence, since it exists to describe population
  statistics, not to power an audit.

# ADR 0005 — Split the human corpus: 4,000-essay baseline vs. 600-essay training pool

## Context

The pipeline needs human text for two different jobs that don't share a
size requirement. Stylometric features (Module 2, 7B) need no model and
cost milliseconds per essay — there's no reason to starve the reference
distribution used for z-scoring (7C). The training pool, by contrast,
goes through the likelihood scorer (Module 4, Qwen2.5-0.5B forward
passes) and becomes actual classifier training data — every essay in it
costs real compute and, more importantly, needs to be ELL-balanced for
the bias audit (C5), which a large uniform sample would dilute.

## Decision

Sample two disjoint pools from the 25,996-essay joined human corpus:
a **4,000-essay baseline** (plain uniform random, `sample_human_pools`
in `src/detector/ingest/sampling.py`) used only to establish "what human
writing looks like" — the reference distribution every feature gets
z-scored against — and a separate **600-essay training pool**,
ELL-balanced (see ADR 0006), that actually flows through the scorer and
becomes classifier training data. No essay appears in both.

## Alternatives considered

- **One shared pool, e.g. 4,000 essays, used for both baseline stats and
  training.** Rejected — training pool essays would then also anchor
  their own z-score reference, and the ELL-balancing needed for the bias
  audit would apply to the baseline too, corrupting it: 4,000 z-score
  reference essays should reflect natural population statistics, not an
  artificially rebalanced sample.
- **One small pool (e.g. 600) used for everything, including the z-score
  baseline.** Rejected — 600 essays is too few to give stable per-feature,
  per-length-band reference distributions; the baseline pool is exactly
  the place where "more data, no compute cost" applies, so there's no
  reason to shrink it to match the training pool's size.
- **Overlapping pools (training essays also counted in the baseline).**
  Rejected — disjointness keeps the z-score reference honest; an essay
  shouldn't help define the very distribution it's later compared
  against.

## Consequences

- Two pools, two purposes, two sizes — `docs/DATA_CARD.md` documents
  both explicitly so nobody downstream assumes they're interchangeable.
- The 4,000-essay baseline can be computed once (Module 2) and reused
  across every later module; the 600-essay training pool is regenerated
  or extended only when the classifier itself needs retraining.
- Sampling both from the same 25,996-essay pool, disjointly, means total
  human coverage used across the project is 4,600 of 25,996 (~18%) —
  documented as a coverage note, not hidden.

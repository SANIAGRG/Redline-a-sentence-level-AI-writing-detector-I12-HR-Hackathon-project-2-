# ADR 0011 — TPR at 1% FPR as the operating point

## Context

A detector like this has an asymmetric cost structure: falsely
accusing a student of using AI carries a different, generally heavier,
cost than missing a machine-written essay. Maximizing accuracy or AUC
treats both error types as equally costly, which they are not for this
use case.

## Decision

Report and tune around **TPR at 1% FPR**, not maximum accuracy or AUC.
Accuracy and AUC are still reported (docs/EVALUATION.md) but as
secondary numbers. The 1% FPR ceiling means: out of every 100 human
essays Redline sees, at most ~1 should be wrongly flagged, and the
headline number is how many actual machine-influenced essays get caught
within that constraint.

## Alternatives considered

- **Maximize accuracy.** Rejected -- with class balance skewed toward
  whichever pool happens to be larger (an artifact of sampling, not the
  real-world base rate), an accuracy-maximizing threshold can quietly
  trade a large increase in false accusations for a small accuracy gain,
  exactly the asymmetry this project should not accept by default.
- **Maximize AUC.** Rejected as the *operating point* (kept as a
  secondary summary metric) -- AUC describes ranking quality across all
  thresholds at once, but says nothing about where the threshold
  actually gets set, which is the decision that matters to a real
  student.
- **A symmetric cost threshold (e.g. Youden's J statistic).** Rejected
  -- still implicitly treats a false accusation and a missed detection
  as equally costly, which the README's own justification for this ADR
  explicitly argues against.

## Consequences

- At the very small evaluation pools this deadline run produced (ADR
  0008 -- e.g. human_training test split on the order of 10 documents),
  a literal 1% FPR point may not be achievable with any real precision;
  `evaluate.py`'s `tpr_at_fpr` picks the tightest achievable operating
  point and documents the actual threshold used, rather than
  interpolating a number the data can't support.
- Reported TPR at this operating point will be noisier and less
  reassuring-looking than an accuracy or AUC headline would be -- that's
  the honest tradeoff of choosing the harder, more relevant metric.
- The threshold is fit once on the in-distribution held-out split, then
  applied unchanged to the held-out-family and modern-generator
  conditions -- so a drop in TPR under those conditions reflects real
  generalization difficulty, not a threshold quietly re-tuned to flatter
  each condition.

# ADR 0010 — Logistic regression over gradient boosting for the shipped model

## Context

The evidence panel (C3) needs every flag to carry a visible, defensible
reason: which measurement fired, how far it sits from baseline, and
its contribution to the score. The model choice has to support that
directly, not just produce an accurate probability.

## Decision

Ship **L2-regularised logistic regression on z-scored features**,
isotonic-calibrated (falls back to sigmoid/Platt calibration when a
pool is too small for isotonic to fit stably -- see `train.py`).
Coefficients read directly as evidence: `contribution = z_score x
coefficient` for each feature, which is exactly what the app's evidence
panel displays, sorted by magnitude. A LightGBM comparison was planned
in the original spec (train both, report both, justify the interpretable
one shipping) but was scoped out of this build's compute budget (ADR
0008, spec's own first-item-to-scope-out) -- the interpretability argument
below was always the primary reason to prefer logistic regression, not
a secondary one built on the comparison, so shipping without the
comparison table doesn't weaken the decision.

## Alternatives considered

- **Gradient boosting (LightGBM), as originally planned to compare
  against.** Higher likely accuracy at this data size, but feature
  contributions come from SHAP values or split-importance approximations,
  not exact linear coefficients -- a real loss of the "read directly as
  evidence" property C3 requires, and one more layer between the number
  on screen and what a human reader can independently verify.
- **A small neural classifier over the raw z-scored features.** Rejected
  for the same interpretability reason, more acutely -- no coefficient
  to point to at all without a post-hoc explainer, and no accuracy
  upside justifies that at this dataset size (189 documents total this
  run, per ADR 0008).
- **No regularisation (plain logistic regression).** Rejected --
  L2 keeps coefficients stable given the small, noisy training pools
  this compute-scoped run produced; an unregularised fit on ~50 training rows
  (human_training + machine_training combined) risks wildly overconfident
  weights on whichever feature happens to separate the tiny sample.

## Consequences

- Every row in the evidence panel is a real, traceable
  `z_score x coefficient` product -- not an approximation, not a
  post-hoc explanation bolted onto an opaque model.
- No LightGBM comparison exists in docs/EVALUATION.md this cycle
  (documented in docs/LIMITATIONS.md) -- the accuracy cost of that
  omission is unmeasured, but the interpretability property this ADR
  protects does not depend on knowing it.
- Isotonic calibration's instability on small pools (handled via a
  sigmoid fallback in `train.py`) is itself a symptom of this build's
  compute-scoped training pools (ADR 0008) -- worth revisiting with
  more data in a future iteration of this project.

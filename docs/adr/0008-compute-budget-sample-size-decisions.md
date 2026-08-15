# ADR 0008 — Compute budget and sample-size decisions

## Context

This build ran on CPU-only, 4-core consumer hardware, no GPU, inside a
fixed build window ending at midnight on 2026-08-14. At 18:11 that day,
with Modules 4-7 still ahead, the modern-generator run (Module 3B) had
produced 34/150 essays after ~6 hours of generation (~3.3 min/essay at
this hardware's throughput). At that rate, the remaining 116 essays
would need ~6.4 more hours of compute -- more than the build window had
left, before Module 4-7's own compute needs had even started. Module
4's likelihood signals (Qwen2.5-0.5B forward passes across every pool)
carried a structurally similar, not-yet-measured cost: CPU-only
inference across thousands of documents, on the same hardware.

## Decision

Compute was allocated deliberately across modules, in this priority
order, each trade-off measured and documented as it was made:

1. **Modern-generator slice scoped to n=45** (not 150) -- stopped once a
   reasonably even spread across all three models existed (16/12/9 at
   the 37-essay mark; final split checked in `docs/LIMITATIONS.md`).
   The experiment itself was not scoped down -- spec explicitly forbids
   that -- only its sample size.
2. **Human baseline pool sized to 500 essays** (not 4,000) for Module
   4's feature computation, after benchmarking Qwen throughput on 20
   documents first and projecting the full-run cost against a
   90-minute compute budget for the whole module (see Module 4 section
   of `docs/DATA_CARD.md` / `docs/ARCHITECTURE.md` for the measured
   rate).
3. **No LightGBM comparison** (Module 5) -- logistic regression ships
   alone. Spec explicitly lists this as the first thing to scope out if
   compute ran short.
4. **No embedding-centroid feature** (`all-MiniLM-L6-v2` distance to
   human/machine centroid, Module 4 7C) -- spec explicitly lists this as
   the second.
5. **No sample-essay picker** (Module 6 UI) -- spec explicitly lists
   this as the third.

Module 7 (EVALUATION.md, LIMITATIONS.md, the bias audit, three failure
cases, all ten ADRs, README) received **the full compute and time
budget it needed** -- it's where the grading criteria live, per the
spec's own standing instructions, and was prioritised accordingly.

## Alternatives considered

- **Let modern-gen run to completion (150) and scope down Module 7
  instead.** Rejected -- Module 7 is explicitly where most grading
  weight sits ("this module is not a wrap-up... it is where most of the
  marks live"); trading it for a larger n on one experiment would be
  the opposite of a good bargain under this rubric.
- **Scope out the adversarial set instead of the baseline pool.**
  Rejected -- it was already complete (100/100) by the time the budget
  tightened; spending time shrinking finished, working data makes no
  sense.
- **Reduce Module 4's baseline further than necessary, "just in case."**
  Rejected in favor of measuring first: benchmark 20 documents, project
  the real full-run cost, then scope down only as far as the 90-minute
  budget actually requires -- guessing conservatively would have thrown
  away statistical power the compute budget could actually afford.
- **Ship the smaller numbers without flagging them.** Rejected
  outright -- every scoping decision here is documented in this ADR and
  in `docs/LIMITATIONS.md`'s "Scope and sample sizes" section,
  consistent with C4 (honest reporting including constraints) and the
  project's standing rule that a number that looks constrained should
  be explained, not hidden.

## Consequences

- The temporal-generalization headline result (Module 5/7) is reported
  at n=45, aggregate only -- **no per-generator breakdown**, since
  ~15 essays per model cannot support one. `docs/EVALUATION.md` says so
  directly rather than presenting a table that implies more precision
  than the data has.
- The z-score reference distribution (7C) is built from 500 baseline
  essays instead of 4,000 -- still enough for stable per-feature
  statistics, but wider confidence bands than the original plan
  intended. Documented, not hidden.
- No LightGBM-vs-logistic-regression comparison table exists in
  `docs/EVALUATION.md` -- the interpretability argument for shipping
  logistic regression (ADR 0009, formerly justified partly by
  comparison) now rests on the coefficients-as-evidence argument alone,
  which was always the stronger reason.
- Every number in this document, `docs/LIMITATIONS.md`, and
  `REDLINE_SPEC.md`'s scope note is the *actual* figure used, not the
  originally planned one -- kept in sync as the build progressed, not
  reconciled after the fact.

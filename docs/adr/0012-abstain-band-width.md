# ADR 0012 — Abstain band and how its width was chosen

## Context

A detector that always answers, even on essays it has weak evidence
about, misrepresents its own confidence. Two cases specifically need an
explicit "insufficient evidence" response rather than a forced verdict:
essays too short for the underlying features to be reliable, and
essays whose score sits close enough to the decision threshold that
the model genuinely can't distinguish the two classes.

## Decision

Two abstain triggers, both implemented in
`src/detector/explain/evidence.py`:

1. **Hard word-count floor.** Essays under 150 words always abstain
   (`MIN_WORDS_FOR_VERDICT`), regardless of score. Below this length,
   stylometric features (MATTR, burstiness, discourse-marker rate, etc.)
   are too sparse to be stable, and likelihood sampling has too few
   sentences to draw a stratified early/mid/late sample from.
2. **Uncertainty band around the probability.** Every non-abstained
   verdict ships with a band, not a bare percentage:
   `width = 0.15 x (1 - 2x|p - 0.5|) + 0.05`. The band is widest (0.20)
   exactly at `p = 0.5`, where the model is least decisive, and narrows
   toward 0.05 as the probability approaches either extreme. This is a
   deliberately simple heuristic, not a value derived from the
   calibration curve's actual local slope (which the spec's original
   plan called for) -- see Consequences.

## Alternatives considered

- **Derive band width from the calibration curve's local slope**, as
  originally planned. Rejected under the deadline (ADR 0008) -- the
  training pools this run produced are too small (human_training 25,
  machine_training 25) for a calibration curve to have a stable local
  slope to derive a width from; a width computed from that curve would
  be noise dressed up as precision.
- **A fixed-width band regardless of probability.** Rejected -- it
  would either be too wide near the extremes (where the model padded
  with more contrast between classes really is more confident) or too
  narrow near 0.5 (where it demonstrably isn't), in either case
  misrepresenting confidence in a way the whole point of showing a band
  is meant to avoid.
- **No abstain band for length, only a probability-based one.**
  Rejected -- a 60-word essay can still land far from the decision
  threshold by chance; length-based abstain catches a failure mode
  (unreliable features) that a probability-based band alone cannot.

## Consequences

- The band-width formula is an explicit, documented heuristic, not the
  calibration-curve-derived width the spec envisioned -- flagged here
  and in `docs/LIMITATIONS.md` so nobody mistakes it for more rigorous
  than it is.
- Once training pools are large enough for a stable calibration curve,
  replacing the heuristic with a curve-derived width is a direct,
  scoped follow-up -- the abstain *mechanism* (word-count floor +
  band around the threshold) does not need to change, only how the
  width is computed.
- The word-count floor is a hard cutoff, not itself calibrated against
  measured feature stability at various lengths -- 150 was chosen as a
  reasonable round number consistent with the spec's own suggested
  floor, not derived from this project's own data.

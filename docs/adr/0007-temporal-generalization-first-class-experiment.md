# ADR 0007 — Temporal generalization as a first-class experiment

## Context

DAIGT-V2 was assembled in late 2023, and every generator family in it
(GPT-3.5, Llama-2, Falcon-180B, Claude v6/v7, PaLM, Cohere Command) is
now previous-generation. Every Kaggle competitor building on this data
trained and tested entirely inside that era, because nothing newer
existed at the time. That leaves an open, answerable question nobody in
that competition could ask: does a detector trained on 2023-era output
still work on 2024-25-era models? Ollama gives local access to
current-generation small models (`llama3.2:3b`, `gemma2:2b`,
`phi3.5:3.8b`) that postdate every DAIGT generator.

## Decision

Generate a dedicated 150-essay modern-generator slice (Module 3B) --
50 essays each from the three current-generation models -- and treat it
as a **first-class, separately-reported test condition** in Module 5
(alongside in-distribution and held-out-family), not folded into
training and not treated as an afterthought. Essays are generated
across four prompt conditions (bare, persona, style-exemplar, evasion)
using PERSUADE's 8 self-contained ("Independent"-task) prompts, cycled
for topic diversity. This slice is held out entirely -- it is Module 5's
evaluation, never training data.

## Alternatives considered

- **Skip the modern-generator slice; report only in-distribution and
  held-out-family results.** Rejected -- this is exactly the limitation
  the brief calls out competitors as unable to see past, and Module 3B's
  own header in the spec calls it "the headline experiment." Skipping it
  would mean building a detector nobody could tell is 2023-era-specific
  or not.
- **Mix modern-generator essays into the training pool instead of
  holding them out.** Rejected -- if the classifier trains on modern
  output, the experiment can no longer ask "does a 2023-trained detector
  generalize to 2024-25 models," because the premise (training data
  predates the generators being tested) no longer holds.
- **Use only one modern model instead of three.** Rejected -- three
  distinct architectures/vendors (Meta's Llama 3.2, Google's Gemma 2,
  Microsoft's Phi-3.5) gives a per-generator breakdown (spec 7A item 7)
  instead of one anecdote; a single-model result wouldn't distinguish
  "detectors decay with newer models in general" from "this particular
  model happens to evade this particular detector."
- **Use all 15 PERSUADE prompts, including the 7 "Text dependent" ones
  that reference a source passage.** Rejected for this module's scope --
  including the source passage as context adds real complexity (context-
  window budgeting under H5's num_ctx=2048, passage-relevance grading)
  for uncertain benefit; the 8 self-contained ("Independent" task)
  prompts already give solid topic diversity (150 essays / 8 topics
  ≈ 19 each) without it. Documented as a scope decision, not an
  oversight -- see docs/DATA_CARD.md.

## Consequences

- Whatever the result, it's reportable: a large accuracy drop on the
  modern slice is a real, defensible finding about detector decay
  (strengthens Module 7's Limitations section); a small drop is evidence
  the underlying signals (likelihood, stylometry) are more
  generator-invariant than expected -- also a real finding. Module 5's
  spec explicitly forbids tuning this drop away.
- The per-generator breakdown (llama3.2:3b vs. gemma2:2b vs. phi3.5:3.8b)
  requires the modern-gen manifest to track which model produced each
  essay, not just "modern vs. not" -- carried through to Module 7's
  evaluation table (7A item 7).
- Because only 8 of PERSUADE's 15 topics are used here, the
  modern-generator slice's topic coverage is narrower than the
  in-distribution training pools (Module 2, which covers all 15). This
  is a deliberate, documented scope difference, not a leak risk --
  Module 5 evaluates this slice as its own held-out condition, never
  compared directly against the training pool's topic distribution the
  way Module 2's topic check compares training pools to each other.

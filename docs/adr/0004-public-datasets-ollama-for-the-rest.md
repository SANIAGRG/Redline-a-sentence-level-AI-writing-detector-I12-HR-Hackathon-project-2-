# ADR 0004 — Public datasets for machine essays; Ollama reserved for what can't be downloaded

## Context

H1 rules out paid APIs entirely. H3/H5 mean local generation is slow
(CPU-only, one model resident at a time) and thermally limited (H4). We
need three kinds of machine-authored text: 2023-era generator output for
training and in-era held-out evaluation, a polished/human-plus-model-edit
corpus, and modern (2024-25) generator output for the temporal experiment.
Generating all of it locally would consume the entire ~6-hour overnight
budget (H3) before Module 4 could even start, and DAIGT-V2 already
contains 2023-era output from 15+ generator families at no compute cost.

## Decision

Machine-authored 2023-era essays come entirely from DAIGT-V2 (already on
disk). Local generation via Ollama is reserved for exactly the two things
that cannot be downloaded from anywhere: the polished/mixed corpus
(Module 3A — nobody has published paragraph-level human-draft-then-model-
polish data with preserved sentence alignment) and the modern-generator
slice (Module 3B — DAIGT predates 2024-era models by construction, so no
public dataset can contain them). The polish runner is written and
started before the end of Module 1 specifically so its ~6-hour budget
overlaps with Module 2's CPU-light feature work instead of blocking it.

## Alternatives considered

- **Generate all machine essays locally for full control over prompts and
  balance.** Rejected — wasteful of the fixed generation budget (H3) on
  data that already exists for free, and DAIGT's 15+ generator families
  give more generator diversity than we could produce overnight anyway.
- **Skip the modern-generator slice and rely solely on DAIGT.** Rejected —
  it would mean training and testing entirely inside 2023-era models, the
  exact limitation the brief calls out competitors as unable to see past;
  Module 3B is explicitly the headline experiment.
- **Use a paid API (e.g. GPT-4o) for the modern-generator slice instead of
  local 2024-25-class Ollama models.** Rejected outright by H1.

## Consequences

- Total local generation is bounded to two runners (polish, modern-gen)
  plus the adversarial set, each resumable and checkpointed (H4) so a
  thermal-throttle death mid-run does not cost more than one batch.
- The project's generator diversity for the *in-era* condition is
  entirely inherited from DAIGT's curation choices — documented as a
  coverage gap in `docs/DATA_CARD.md` and `docs/LIMITATIONS.md`, not
  something we control.
- Because the polish runner is the longest local job and has no public
  substitute, it is the first thing kicked off in Module 1 rather than
  scheduled later — see H3/Module 1 exit criteria.

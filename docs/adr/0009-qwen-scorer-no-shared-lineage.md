# ADR 0009 — Qwen2.5-0.5B as scorer, sharing lineage with no generator

## Context

The likelihood signals (7A) need a local model to extract token
log-probabilities, entropy, and cross-model divergence from essay text.
If that scorer shares architecture/training lineage with one of the
essay generators being detected, its signal on that generator's output
is inflated for the wrong reason -- it's recognizing family resemblance,
not measuring genuine statistical anomaly -- and that inflation quietly
corrupts held-out results without any visible warning sign.

## Decision

Use **Qwen2.5-0.5B base + Qwen2.5-0.5B-Instruct**, loaded via
`transformers` (not Ollama -- raw logits are required, which Ollama
doesn't expose). Qwen is absent from every generator set in this
project: DAIGT-V2's 15 families (GPT-3.5, Llama-2, Falcon-180B, Claude
v6/v7, PaLM, Cohere Command, various Mistral/Llama fine-tunes) and the
three locally-generated modern models (`llama3.2:3b`, `gemma2:2b`,
`phi3.5:3.8b`). No shared lineage with anything being scored, in either
the 2023-era or modern-generator condition.

## Alternatives considered

- **Use one of the generation models itself as scorer** (e.g. reuse
  `llama3.2:3b`, already loaded for Module 3). Rejected outright --
  scoring Llama-generated text with a Llama-family scorer is exactly the
  inflation problem this ADR exists to avoid, and it's the modern-
  generator condition specifically that the temporal experiment (ADR
  0007) depends on being clean.
- **A larger, more capable open model (e.g. a 7B-class model) as
  scorer.** Rejected on hardware grounds -- H5 and the CPU-only laptop
  budget favor a small model, and 0.5B is enough to produce a usable
  probability distribution for log-prob/entropy/rank purposes without
  the latency cost a larger model would add across every sentence of
  every essay.
- **GPT-2 or another older, well-known perplexity-baseline model.**
  Rejected -- meaningfully worse language modeling quality than a 2024-
  era small model, which would weaken every likelihood signal, for no
  lineage benefit over Qwen (GPT-2 also shares no lineage with the
  generators, but its perplexity estimates are simply less reliable).

## Consequences

- Cross-perplexity ratio (Binoculars-style, base-vs-instruct) is
  well-defined precisely because both halves come from the same Qwen
  family but differ only in instruction-tuning -- a clean base/instruct
  contrast that wouldn't exist if the two scorer halves came from
  unrelated models.
- Because Qwen has no lineage overlap with any generator, a strong
  likelihood signal on, say, GPT-3.5 output is attributable to genuine
  distributional difference from human writing, not scorer/generator
  family resemblance -- this is the load-bearing assumption behind
  every held-out result in docs/EVALUATION.md.
- Under the deadline compute crunch (ADR 0008), Qwen2.5-0.5B's forward-
  pass cost was still the binding constraint that forced sentence
  sampling and pool-size cuts -- a larger scorer would have made an
  already-tight budget infeasible.

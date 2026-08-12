# ADR 0001 — Local models are measurement instruments, never judges

## Context

The obvious way to build "does this essay look AI-written" is to prompt a
capable chat model with the essay and ask it to decide. The hackathon brief
explicitly anticipates this and says a wrapper of that shape will be marked
down: general-purpose chat models can already do reasonable zero-shot
AI-text detection, so a prompt-and-parse wrapper adds no evidence, no
calibration, and no ablation story of its own. It is also unfalsifiable —
there is no way to show *why* the model said what it said.

## Decision

No model in the runtime path is ever given a prompt resembling "is this
AI-written?" or asked for a verdict. Local models (Qwen2.5-0.5B base and
instruct) are run in teacher-forced mode over the essay text to produce
*numbers*: token log-probabilities, predictive entropy, cross-model
divergence. A separate classifier we train ourselves (logistic regression
on z-scored features) consumes those numbers and produces the score. The
language model never sees a question; it only ever sees the essay text
and returns a distribution over next tokens.

## Alternatives considered

- **Prompt a chat model for a verdict, optionally with chain-of-thought.**
  Rejected — fails C1 directly, produces no per-sentence evidence, and the
  brief singles this pattern out as insufficient.
- **Fine-tune a chat model as a classifier head.** Rejected for this
  hardware budget (no GPU) and because it still produces an opaque score
  rather than named, inspectable signals.
- **Ensemble a chat-model verdict with our own features as one input among
  many.** Rejected — even as one signal among several, it reintroduces an
  unauditable component and the risk that the classifier learns to just
  trust it.

## Consequences

- Every score is traceable to a named measurement (log-prob, entropy,
  curvature, a stylometric count) with a value and a baseline range —
  this is what makes the evidence panel (C3) possible at all.
- We give up whatever zero-shot accuracy a large chat model might have on
  obviously-machine text, in exchange for an interpretable, ablatable
  pipeline.
- Feature engineering (Module 4) becomes the highest-leverage work in the
  project, not prompt engineering.

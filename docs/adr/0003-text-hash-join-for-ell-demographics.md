# ADR 0003 — Join DAIGT to PERSUADE on normalised-text hash, not ID

## Context

DAIGT-V2's human rows (`source == "persuade_corpus"`, 25,996 rows) are
literal PERSUADE 2.0 essays, but DAIGT dropped every demographic column,
including `ell_status`, which the bias audit (C5) needs. PERSUADE 2.0
(25,996 rows) has the demographics but was released independently, and
its `essay_id_comp` identifiers do not appear in DAIGT at all — there is
no shared key to join on directly.

## Decision

Join on a hash of normalised essay text: strip leading/trailing
whitespace, collapse internal whitespace runs to a single space,
lowercase, then SHA-256 the result (`detector/ingest/normalisation.py`).
Both sides are hashed and merged on that key. Where PERSUADE contains
duplicate normalised text (a small number of boilerplate/empty
submissions), the first occurrence is kept so the join stays 1:1.

## Alternatives considered

- **Join on raw, unnormalised text equality.** Rejected — any difference
  in trailing whitespace, line-ending style, or double-spacing between
  the two CSV exports would silently drop a match. Text hashing without
  normalisation is exactly the "IDs differ between releases" trap in a
  different form.
- **Fuzzy/approximate text matching (e.g. Levenshtein or embedding
  similarity above a threshold).** Rejected as unnecessary complexity —
  normalised exact match already achieves a 100% match rate (25,996 /
  25,996) on this data, so there is no residual gap to close with fuzzy
  matching, and fuzzy matching would need a threshold decision with no
  ground truth to tune it against.
- **Match on `(prompt_name, word_count)` composite key.** Rejected —
  many essays share a prompt and a similar length; this key is not
  unique enough and would introduce silent mismatches.

## Consequences

- The join achieved a 100% match rate and 2,244 usable ELL-flagged
  essays, comfortably above the ~200 minimum set as the go/no-go
  threshold for downloading the ELLIPSE corpus. ELLIPSE is not needed.
- The join is deterministic and re-runnable (`make join`) — no manual
  reconciliation step.
- This approach is specific to the fact that PERSUADE text is largely
  unmodified when it flows into DAIGT. If a future dataset introduces
  even light text normalisation upstream (e.g. Unicode quote
  substitution), the hash join would need a matching normalisation step
  added here, not a different join strategy.

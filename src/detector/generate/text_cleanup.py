"""Strip conversational preamble that small instruct models sometimes
prepend to an otherwise-correct response ("Here's the revised
paragraph:\n\n...") despite being told to return only the text itself.

Found during the Module 3 sentence-alignment spot-check: 55/1416 (3.9%)
of polish-corpus revisions had this contamination. Left unstripped, the
embedded blank line collides with the paragraph-boundary delimiter used
everywhere else in the pipeline, corrupting document reconstruction --
not just cosmetic. Applied inside OllamaClient.generate() so every
future generation call is covered, not just a one-off cleanup.
"""

from __future__ import annotations

import re

_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"here.{0,15}(?:is|s|are).{0,30}(?:revised|rewritten|corrected|paragraph|version)"
    r"|sure[,!]"
    r"|i.ve (?:revised|rewritten|corrected)"
    r"|certainly"
    r"|\*\*revised"
    r").*?\n\n",
    re.IGNORECASE | re.DOTALL,
)


def strip_llm_preamble(text: str) -> str:
    """Remove a leading conversational preamble up to the first blank
    line, if the text before it looks like one. Leaves everything else
    untouched -- if there's no blank-line-delimited preamble, returns the
    input unchanged.
    """
    match = _PREAMBLE_RE.match(text)
    if match:
        return text[match.end() :].strip()
    return text

"""Text normalisation used as the join key between DAIGT and PERSUADE.

DAIGT-V2's human rows are PERSUADE essays with the demographic columns
stripped, and the two releases do not share essay IDs. We join on a
normalised-text hash instead.
"""

from __future__ import annotations

import hashlib
import re

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Strip whitespace, collapse internal spaces, lowercase."""
    collapsed = _WHITESPACE_RE.sub(" ", text.strip())
    return collapsed.lower()


def text_hash(text: str) -> str:
    """Stable hash of normalised text, used as the join key."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()

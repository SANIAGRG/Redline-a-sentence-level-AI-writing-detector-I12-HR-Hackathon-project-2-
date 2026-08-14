"""Adversarial attacks over existing machine essays (Module 3C).
Eval-only -- never trained on. No Ollama (H2 reserves it for the polish
corpus and the modern-generator slice only), so both attacks here are
downloaded/local-library based: WordNet synonym substitution for the
paraphrase attack, pure character-level perturbation for typo injection.

Both are standard, well-documented baselines in the adversarial-text
literature (synonym-substitution attacks e.g. TextFooler/PWWS; keyboard-
adjacent typo injection for OCR/typing-error robustness) -- not full
paraphrase-model generation, which would need Ollama and blow past H2's
two-job budget for the sake of an eval-only set the spec explicitly
ranks as the first thing to cut if time is short.
"""

from __future__ import annotations

import random
import string

from nltk.corpus import wordnet as wn

from detector.ingest.segmentation import get_nlp

# QWERTY physical-adjacency map, lowercase only -- used for both
# insertion (insert a neighbour) and substitution (replace with a
# neighbour) typos, since those are the errors a real typist makes.
_KEYBOARD_ADJACENCY: dict[str, str] = {
    "q": "wa",
    "w": "qesa",
    "e": "wrsd",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}

_POS_TO_WORDNET = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}


def _typo_word(word: str, rng: random.Random) -> str:
    if len(word) < 3:
        return word
    op = rng.choice(["swap", "delete", "insert", "substitute"])
    i = rng.randrange(len(word))
    lower = word.lower()

    if op == "swap" and len(word) >= 4:
        j = min(i + 1, len(word) - 1)
        chars = list(word)
        chars[i], chars[j] = chars[j], chars[i]
        return "".join(chars)
    if op == "delete":
        return word[:i] + word[i + 1 :]
    if op == "insert":
        neighbours = _KEYBOARD_ADJACENCY.get(lower[i], string.ascii_lowercase)
        return word[:i] + rng.choice(neighbours) + word[i:]
    if op == "substitute":
        neighbours = _KEYBOARD_ADJACENCY.get(lower[i], string.ascii_lowercase)
        return word[:i] + rng.choice(neighbours) + word[i + 1 :]
    return word


def inject_typos(text: str, rate: float, seed: int) -> str:
    """Introduce a keyboard-plausible typo into ~rate of alphabetic words
    (swap/delete/insert/substitute a character), leaving punctuation and
    spacing untouched.
    """
    rng = random.Random(seed)
    tokens = text.split(" ")
    out = []
    for token in tokens:
        core = token.strip(string.punctuation)
        if core.isalpha() and rng.random() < rate:
            typoed_core = _typo_word(core, rng)
            out.append(token.replace(core, typoed_core, 1))
        else:
            out.append(token)
    return " ".join(out)


def _best_synonym(word: str, wn_pos: str) -> str | None:
    synsets = wn.synsets(word, pos=wn_pos)
    for synset in synsets:
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ")
            if candidate.lower() != word.lower() and candidate.isalpha():
                return candidate
    return None


def paraphrase_synonyms(text: str, rate: float, seed: int) -> str:
    """Replace ~rate of eligible content words (NOUN/VERB/ADJ/ADV, not
    proper nouns) with a same-POS WordNet synonym. A standard
    synonym-substitution paraphrase-attack baseline.
    """
    rng = random.Random(seed)
    nlp = get_nlp()
    doc = nlp(text)

    out_parts: list[str] = []
    for token in doc:
        wn_pos = _POS_TO_WORDNET.get(token.pos_)
        if wn_pos and rng.random() < rate:
            synonym = _best_synonym(token.text, wn_pos)
            if synonym:
                replacement = synonym.capitalize() if token.text[:1].isupper() else synonym
                out_parts.append(replacement + token.whitespace_)
                continue
        out_parts.append(token.text_with_ws)
    return "".join(out_parts)

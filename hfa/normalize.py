"""Text normalization, applied identically to the quote and to the source.

Steps (see docs in README "Normalization"):
 1. HTML entity decoding (html.unescape).
 2. Unicode NFKC (compatibility composition: ligatures, full-width forms,
    NBSP variants collapse to their plain equivalents). NFKC rather than NFC
    because web pages mix compatibility forms that a human reads as the same
    text; NFKC is a superset of NFC's canonical composition.
 3. Punctuation unification: curly quotes -> straight, en/em dash and minus
    -> '-', ellipsis char -> '...'.
 4. Deletion of soft hyphen (U+00AD) and zero-width characters.
 5. Whitespace collapse: any run of whitespace -> one space; trimmed.
 6. casefold() for comparison only (fold()). Display keeps the case.

NOT normalized on purpose: digits, currency symbols, negations, footnote
markers such as "[3]". A quote must match those exactly.
"""
from __future__ import annotations

import html
import re
import unicodedata

_PUNCT = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    "…": "...",
}
_DELETE = {"­", "​", "‌", "‍", "⁠", "﻿"}
_TRANS = str.maketrans({**_PUNCT, **{c: None for c in _DELETE}})
_WS = re.compile(r"\s+")


def norm(text: str) -> str:
    """Display-preserving normalization (case kept)."""
    if not text:
        return ""
    t = html.unescape(text)
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(_TRANS)
    t = _WS.sub(" ", t)
    return t.strip()


def fold_with_map(normed: str) -> tuple[str, list[int]]:
    """casefold() a normalized string, returning a map from each folded
    character index back to the index in `normed` (casefold may change the
    length, e.g. German sharp s -> 'ss')."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(normed):
        f = ch.casefold()
        out.append(f)
        idx.extend([i] * len(f))
    return "".join(out), idx


def fold(text: str) -> str:
    return norm(text).casefold()

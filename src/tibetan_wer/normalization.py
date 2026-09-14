"""Text normalization, made explicit.

Error rates depend on conventions that are easy to leave implicit: whether text
is Unicode-normalized, whether variant punctuation is folded together, whether
whitespace is trimmed. Two systems scored under different conventions are not
comparable, which is a known source of incomparable CER figures in the OCR
literature (Neudecker et al. 2021).

This package normalizes nothing by default -- text is scored as stored. Pass
``normalize=True`` to a metric, or call :func:`normalize` yourself, to apply
the conventions below, and say which you used when reporting.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable

__all__ = ["normalize", "normalize_all", "TSEK", "TSEK_BSTAR", "SHAD", "SBRUL_SHAD"]

TSEK = "་"        # ་  intersyllabic tsek
TSEK_BSTAR = "༌"  # ༌  tsek bstar, a non-breaking variant
SHAD = "།"        # །  shad, the clause delimiter
SBRUL_SHAD = "༈"  # ༈  sbrul shad, a section-opening variant


def normalize(
    text: str,
    form: str = "NFC",
    fold_tsek_bstar: bool = True,
    fold_sbrul_shad: bool = True,
    strip: bool = True,
) -> str:
    """Apply the conventions above to one string.

    Parameters
    ----------
    form : {'NFC', 'NFD', 'NFKC', 'NFKD', None}
        Unicode normalization form. Tibetan stacks are sequences of codepoints
        rather than precomposed characters, so this changes less than it does
        in many scripts, but it does settle vowel-sign ordering.
    fold_tsek_bstar : bool
        Rewrite tsek bstar (U+0F0C) as tsek (U+0F0B). Affects SER, which splits
        on the latter only.
    fold_sbrul_shad : bool
        Rewrite sbrul shad (U+0F08) as shad (U+0F0D).
    strip : bool
        Trim surrounding whitespace.

    Notes
    -----
    Folding is a scoring decision, not a claim that the marks are equivalent:
    it stops a system being penalized for a distinction the reference makes
    inconsistently. Do not fold when the distinction is what you are measuring.
    """
    if form:
        text = unicodedata.normalize(form, text)
    if fold_tsek_bstar:
        text = text.replace(TSEK_BSTAR, TSEK)
    if fold_sbrul_shad:
        text = text.replace(SBRUL_SHAD, SHAD)
    if strip:
        text = text.strip()
    return text


def normalize_all(texts: Iterable[str], **kwargs) -> list:
    """:func:`normalize` over an iterable of strings."""
    return [normalize(t, **kwargs) for t in texts]

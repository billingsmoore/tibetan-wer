"""Levenshtein alignment over unit sequences, with the individual edit
operations exposed.

Two backends compute the same distance:

``python``
    The reference implementation: a dynamic-programming table with an explicit
    backtrace whose tie-break order is match, substitution, insertion,
    deletion. This is the alignment behind every number published with the SWER
    paper.
``fast``
    ``rapidfuzz`` (or ``python-Levenshtein``), roughly a thousand times
    quicker. It returns the **same edit distance on every input**, so error
    rates are identical to the digit; but where several alignments have equal
    cost it may pick a different one, so the split of a given total into
    substitutions, insertions and deletions can differ by a few percent.

Hence the default: rates use ``fast`` when it is installed, because the result
is provably unchanged, while anything reporting operation counts or individual
operations uses ``python`` unless asked otherwise.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np

__all__ = [
    "EditOperation",
    "available_backends",
    "resolve_backend",
    "edit_matrix",
    "align",
    "edit_counts",
    "distance",
]


class EditOperation(NamedTuple):
    """A single edit in the alignment of a reference to a hypothesis.

    ``op`` is ``"S"``, ``"I"`` or ``"D"``. For an insertion ``ref`` is ``None``
    and ``ref_index`` is the reference position the unit was inserted before;
    for a deletion ``hyp`` is ``None`` and ``hyp_index`` is the corresponding
    hypothesis position. Indices are into the unit sequences, not into the
    original strings.
    """

    op: str
    ref: object
    hyp: object
    ref_index: int
    hyp_index: int


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_TAGS = {"replace": "S", "insert": "I", "delete": "D"}

_fast_module = None
_fast_name = None
try:  # pragma: no cover - depends on what is installed
    from rapidfuzz.distance import Levenshtein as _fast_module  # type: ignore

    _fast_name = "rapidfuzz"
except ImportError:  # pragma: no cover
    try:
        import Levenshtein as _fast_module  # type: ignore

        _fast_name = "python-Levenshtein"
    except ImportError:
        _fast_module = None


def available_backends() -> list[str]:
    """Backends usable in this environment, fastest first."""
    return (["fast"] if _fast_module is not None else []) + ["python"]


def resolve_backend(backend: str = "auto", *, need_operations: bool = True) -> str:
    """Turn ``"auto"`` into a concrete backend name.

    With ``need_operations=False`` the caller wants only a distance, where the
    two backends agree exactly, so the fast one is chosen when available. With
    ``need_operations=True`` the choice affects how a total is split into S/I/D,
    so ``"auto"`` stays on the reference implementation.
    """
    if backend not in ("auto", "fast", "python"):
        raise ValueError(f"unknown backend {backend!r}; expected 'auto', 'fast' or 'python'")
    if backend == "fast":
        if _fast_module is None:
            raise ImportError(
                "backend='fast' requires rapidfuzz or python-Levenshtein: "
                "pip install 'tibetan-wer[fast]'"
            )
        return "fast"
    if backend == "python":
        return "python"
    if not need_operations and _fast_module is not None:
        return "fast"
    return "python"


def backend_library() -> str | None:
    """Name of the library providing the fast backend, or None."""
    return _fast_name


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def edit_matrix(ref_units: Sequence, hyp_units: Sequence) -> np.ndarray:
    """Full Levenshtein cost matrix, ``(len(ref) + 1, len(hyp) + 1)``."""
    r_len, p_len = len(ref_units), len(hyp_units)
    d = np.zeros((r_len + 1, p_len + 1), dtype=np.int32)
    d[:, 0] = np.arange(r_len + 1)
    d[0, :] = np.arange(p_len + 1)
    for i in range(1, r_len + 1):
        for j in range(1, p_len + 1):
            if ref_units[i - 1] == hyp_units[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )
    return d


def _align_python(ref_units: Sequence, hyp_units: Sequence) -> list:
    d = edit_matrix(ref_units, hyp_units)
    i, j = len(ref_units), len(hyp_units)
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_units[i - 1] == hyp_units[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            ops.append(EditOperation("S", ref_units[i - 1], hyp_units[j - 1], i - 1, j - 1))
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ops.append(EditOperation("I", None, hyp_units[j - 1], i, j - 1))
            j -= 1
        else:
            ops.append(EditOperation("D", ref_units[i - 1], None, i - 1, j))
            i -= 1
    ops.reverse()
    return ops


def _align_fast(ref_units: Sequence, hyp_units: Sequence) -> list:
    ops = []
    for tag, i, j in _fast_module.editops(list(ref_units), list(hyp_units)):
        op = _TAGS[tag]
        if op == "S":
            ops.append(EditOperation("S", ref_units[i], hyp_units[j], i, j))
        elif op == "I":
            ops.append(EditOperation("I", None, hyp_units[j], i, j))
        else:
            ops.append(EditOperation("D", ref_units[i], None, i, j))
    return ops


# ---------------------------------------------------------------------------
# Public
# ---------------------------------------------------------------------------

def align(ref_units: Sequence, hyp_units: Sequence, backend: str = "auto") -> list:
    """Return the edit operations turning ``ref_units`` into ``hyp_units``.

    Operations are in left-to-right order; matches are not included.

    >>> [op.op for op in align(list("abc"), list("axc"))]
    ['S']
    """
    if resolve_backend(backend, need_operations=True) == "fast":
        return _align_fast(ref_units, hyp_units)
    return _align_python(ref_units, hyp_units)


def edit_counts(
    ref_units: Sequence, hyp_units: Sequence, backend: str = "auto"
) -> tuple:
    """``(substitutions, insertions, deletions)`` for one pair of sequences."""
    ops = align(ref_units, hyp_units, backend=backend)
    S = sum(1 for o in ops if o.op == "S")
    I = sum(1 for o in ops if o.op == "I")
    D = sum(1 for o in ops if o.op == "D")
    return S, I, D


def distance(ref_units: Sequence, hyp_units: Sequence, backend: str = "auto") -> int:
    """Levenshtein distance between two unit sequences.

    Identical whichever backend runs, so ``"auto"`` takes the fast one when it
    is installed.
    """
    if resolve_backend(backend, need_operations=False) == "fast":
        return int(_fast_module.distance(list(ref_units), list(hyp_units)))
    return int(edit_matrix(ref_units, hyp_units)[len(ref_units)][len(hyp_units)])

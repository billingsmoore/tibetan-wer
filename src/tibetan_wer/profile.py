"""Error-structure diagnostics.

A rate says how much a system is wrong; these say how. All of them are counts
over the alignment, so they cost one extra pass and no extra model:

*Operation inventory*
    Substitutions, insertions and deletions as shares of all edits. Systems at
    the same rate can differ sharply here -- a CTC decoder that drops units and
    an autoregressive decoder that invents them are not equally repairable.
*Boundary edits*
    The share of character edits falling on the tsek itself. These change the
    number of syllables, so they are the character-level errors that move the
    syllable and word rates most.
*Space-producing edits*
    Insertions of U+0020 and substitutions producing it. A space between two
    Tibetan letters is one character edit and can split a syllable in two, so
    over-producing spaces costs more at the word level than at the character
    level. Space is legitimate in Tibetan text, so this measures calibration,
    not illegal output.
*Confusions*
    The most frequent (reference unit -> hypothesis unit) pairs.
*Malformed share*
    With a ``vocabulary``, the share of syllable substitutions and insertions
    producing a syllable that does not occur in it -- the real-word/non-word
    distinction. A malformed syllable is mechanically detectable and so
    correctable without a language model; a well-formed but wrong one is not.
*Concentration*
    How unevenly error is distributed over sentences. A mean rate hides the
    difference between a system that is uniformly mediocre and one that is
    perfect on most lines and catastrophic on a few.
"""
from __future__ import annotations

import collections
from typing import Iterable, Sequence

import numpy as np

from .alignment import align
from .metrics import _apply_normalization, _validate_inputs
from .segmentation import TSEK, segment_all, syllable_segment

__all__ = ["error_profile", "concentration", "gini"]

SPACE = " "


# ---------------------------------------------------------------------------
# Concentration
# ---------------------------------------------------------------------------

def gini(values: Sequence[float]) -> float:
    """Gini coefficient of a non-negative array; 0 is uniform, 1 maximally concentrated."""
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    v = v[v > 0]
    if len(v) < 2 or v.sum() == 0:
        return float("nan")
    v = np.sort(v)
    n = len(v)
    index = np.arange(1, n + 1)
    return float((2 * (index * v).sum()) / (n * v.sum()) - (n + 1) / n)


def concentration(per_utterance: Sequence[float]) -> dict:
    """How unevenly error falls across sentences.

    Returns ``zero_error_share`` (sentences with no error), ``p90`` (90th
    percentile rate), ``worst_decile_share`` (share of all summed per-sentence
    error contributed by the worst tenth of sentences) and ``gini`` (over
    sentences with non-zero error).
    """
    v = np.asarray(per_utterance, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {
            "zero_error_share": float("nan"),
            "p90": float("nan"),
            "worst_decile_share": float("nan"),
            "gini": float("nan"),
            "num_sentences": 0,
        }
    ordered = np.sort(v)[::-1]
    cut = max(1, int(round(0.1 * len(v))))
    total = v.sum()
    return {
        "zero_error_share": float((v == 0).mean()),
        "p90": float(np.percentile(v, 90)),
        "worst_decile_share": float(ordered[:cut].sum() / total) if total > 0 else float("nan"),
        "gini": gini(v),
        "num_sentences": int(len(v)),
    }


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

def _units(texts: Iterable[str], level: str, **segment_kwargs):
    if level == "char":
        return [list(t) for t in texts]
    if level == "syllable":
        return [syllable_segment(t) for t in texts]
    return segment_all(texts, method=level, **segment_kwargs)


def error_profile(
    predictions,
    references,
    level: str = "char",
    vocabulary: Iterable[str] | None = None,
    top_confusions: int = 10,
    normalize=False,
    backend: str = "auto",
    **segment_kwargs,
) -> dict:
    """Structural diagnostics over the alignment of a system's output.

    Parameters
    ----------
    predictions, references : str or sequence of str
    level : {'char', 'syllable', 'botok', 'bert', 'gemini'}
        Unit to align over. Boundary and space diagnostics are character-level
        and are reported only for ``'char'``.
    vocabulary : iterable of str, optional
        Syllables treated as attested, enabling ``malformed`` at
        ``level='syllable'``. Build it from the reference corpus --
        ``set(itertools.chain.from_iterable(syllable_segment(r) for r in refs))``.
        Whitespace is stripped from a syllable before lookup.
    top_confusions : int
        How many (reference -> hypothesis) pairs to report.
    normalize, backend
        As :func:`tibetan_wer.metrics.cer`. ``backend`` defaults to the
        reference alignment, since these counts depend on which operations are
        reported.

    Returns
    -------
    dict
        ``operations`` (counts and shares), ``confusions``, ``concentration``,
        and, where they apply, ``boundary_edits``, ``space_edits``,
        ``malformed``.

    Notes
    -----
    Shares are over all edits in the corpus, micro-averaged, matching how the
    rates themselves are pooled.
    """
    predictions, references = _validate_inputs(predictions, references)
    predictions = _apply_normalization(predictions, normalize)
    references = _apply_normalization(references, normalize)

    ref_units = _units(references, level, **segment_kwargs)
    hyp_units = _units(predictions, level, **segment_kwargs)

    counts = collections.Counter()
    confusions = collections.Counter()
    boundary = space = 0
    malformed = attested = 0
    rates = []
    vocab = None
    if vocabulary is not None:
        vocab = {v.strip() for v in vocabulary}

    for ref, hyp in zip(ref_units, hyp_units):
        ops = align(ref, hyp, backend=backend)
        for op in ops:
            counts[op.op] += 1
            if op.op == "S":
                confusions[(op.ref, op.hyp)] += 1
            if level == "char":
                if op.ref == TSEK or op.hyp == TSEK:
                    boundary += 1
                if op.hyp == SPACE:
                    space += 1
            if vocab is not None and level == "syllable" and op.op in ("S", "I"):
                if op.hyp is not None and op.hyp.strip() in vocab:
                    attested += 1
                else:
                    malformed += 1
        rates.append(len(ops) / len(ref) if ref else float("nan"))

    total = sum(counts.values())

    def share(n):
        return float(n / total) if total else float("nan")

    out = {
        "level": level,
        "num_sentences": len(references),
        "operations": {
            "substitutions": counts["S"],
            "insertions": counts["I"],
            "deletions": counts["D"],
            "total": total,
            "substitution_share": share(counts["S"]),
            "insertion_share": share(counts["I"]),
            "deletion_share": share(counts["D"]),
        },
        "confusions": [
            {"reference": r, "hypothesis": h, "count": c}
            for (r, h), c in confusions.most_common(top_confusions)
        ],
        "concentration": concentration(rates),
    }
    if level == "char":
        out["boundary_edits"] = {"count": boundary, "share": share(boundary)}
        out["space_edits"] = {"count": space, "share": share(space)}
    if vocab is not None and level == "syllable":
        produced = malformed + attested
        out["malformed"] = {
            "malformed": malformed,
            "attested": attested,
            "share": float(malformed / produced) if produced else float("nan"),
            "vocabulary_size": len(vocab),
        }
    return out

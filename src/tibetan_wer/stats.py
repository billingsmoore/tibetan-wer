"""Confidence intervals and system comparison.

Two ways to ask whether two systems differ:

*Do their intervals overlap?* -- :func:`bootstrap_ci` on each, then
:func:`cis_overlap` or :func:`significance_matrix`. This is the criterion used
in the SWER paper, and it is what to use when the systems were scored on
different data, or when all you have is published intervals.

*Is the difference itself distinguishable from zero?* -- :func:`compare`, a
paired test over the per-sentence differences. Both systems saw the same
sentences, and pairing removes the sentence-to-sentence variance that dominates
either interval on its own. Non-overlapping intervals imply a significant
paired difference, but not the reverse: the overlap test is the conservative
one, and it misses real differences that the paired test detects.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "bootstrap_ci",
    "cis_overlap",
    "significance_matrix",
    "compare",
]


# ---------------------------------------------------------------------------
# Extracting per-sentence quantities from whatever the caller passes
# ---------------------------------------------------------------------------

def _per_sentence_arrays(scores):
    """Return ``(rates, edits, reference_lengths)``; the last two may be None.

    Accepts a metric result dict or a bare array of per-sentence rates.
    """
    if isinstance(scores, Mapping):
        rates = np.asarray(scores["per_utterance"], dtype=float)
        sentences = scores.get("per_sentence")
        if sentences is None:
            return rates, None, None
        edits = np.array(
            [np.nan if s is None else s["edits"] for s in sentences], dtype=float
        )
        lengths = np.array(
            [np.nan if s is None else s["reference_length"] for s in sentences], dtype=float
        )
        return rates, edits, lengths
    return np.asarray(scores, dtype=float), None, None


def _require_micro(edits, lengths):
    if edits is None or lengths is None:
        raise ValueError(
            "statistic='micro' needs per-sentence edit counts: pass the metric "
            "result dict itself, not just its 'per_utterance' array"
        )


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    scores,
    n_iterations: int = 10_000,
    percentiles: tuple = (2.5, 97.5),
    seed: int = 42,
    statistic: str = "macro",
    chunk_size: int = 500,
) -> tuple:
    """Bootstrap a rate: ``(mean, lo, hi)``.

    Parameters
    ----------
    scores : dict or array
        A metric result dict, or its ``per_utterance`` array. The micro
        statistic needs the dict.
    statistic : {'macro', 'micro'}
        ``'macro'`` resamples sentences and takes the mean of their rates.
        ``'micro'`` resamples sentences and recomputes total edits over total
        reference length, so long sentences carry proportionally more weight --
        the same weighting as the reported ``micro_*`` rate.
    n_iterations, percentiles, seed
        ``seed`` makes the interval reproducible; the defaults match the papers.
    chunk_size : int
        Replicates per vectorized batch, bounding peak memory on large test
        sets.

    Notes
    -----
    Sentences with no defined rate -- an empty reference, or one dropped by the
    Gemini repetition filter -- are excluded before resampling.
    """
    if statistic not in ("macro", "micro"):
        raise ValueError("unknown statistic {!r}; expected 'macro' or 'micro'".format(statistic))
    rates, edits, lengths = _per_sentence_arrays(scores)
    keep = ~np.isnan(rates)
    if statistic == "micro":
        _require_micro(edits, lengths)
        edits, lengths = edits[keep], lengths[keep]
        point = float(edits.sum() / lengths.sum()) if lengths.sum() else float("nan")
    else:
        rates = rates[keep]
        point = float(rates.mean()) if len(rates) else float("nan")
    n = int(keep.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    replicates = np.empty(n_iterations, dtype=float)
    done = 0
    while done < n_iterations:
        size = min(chunk_size, n_iterations - done)
        idx = rng.integers(0, n, size=(size, n))
        if statistic == "micro":
            replicates[done:done + size] = edits[idx].sum(axis=1) / lengths[idx].sum(axis=1)
        else:
            replicates[done:done + size] = rates[idx].mean(axis=1)
        done += size
    lo, hi = np.percentile(replicates, percentiles)
    return point, float(lo), float(hi)


def cis_overlap(ci_a: tuple, ci_b: tuple) -> bool:
    """True if two ``(lo, hi)`` intervals overlap."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    return not (hi_a < lo_b or hi_b < lo_a)


def significance_matrix(system_cis: Mapping) -> dict:
    """Pairwise non-overlap test over ``{system: (mean, lo, hi)}``.

    Returns nested dicts where ``out[a][b]`` is True when the two systems'
    intervals do not overlap. Conservative: see :func:`compare` for the paired
    alternative.
    """
    names = list(system_cis)
    out = {a: {} for a in names}
    for a in names:
        for b in names:
            _, lo_a, hi_a = system_cis[a]
            _, lo_b, hi_b = system_cis[b]
            out[a][b] = (a != b) and not cis_overlap((lo_a, hi_a), (lo_b, hi_b))
    return out


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def compare(
    result_a,
    result_b,
    n_iterations: int = 10_000,
    seed: int = 42,
    statistic: str = "macro",
    percentiles: tuple = (2.5, 97.5),
    method: str = "both",
    chunk_size: int = 500,
) -> dict:
    """Paired comparison of two systems scored on the same sentences.

    Parameters
    ----------
    result_a, result_b : dict or array
        Metric results for the two systems, aligned sentence by sentence.
    statistic : {'macro', 'micro'}
        Which rate the difference is taken on. ``'micro'`` needs result dicts.
    method : {'both', 'bootstrap', 'permutation'}
        ``'bootstrap'`` resamples sentences (with replacement) and reports a
        confidence interval on the difference, plus the share of replicates
        falling on the other side of zero. ``'permutation'`` swaps the two
        systems' scores within randomly chosen sentences, which is the exact
        null of "these two systems are interchangeable", and reports a p-value.

    Returns
    -------
    dict
        ``rate_a``, ``rate_b``, ``delta`` (a minus b, so negative means ``a``
        is better), ``ci``, ``p_bootstrap``, ``p_permutation``, ``n_paired``,
        ``n_dropped``, ``statistic``, ``significant`` (the interval excludes
        zero).

    Notes
    -----
    Only sentences scored by both systems are used; the rest are counted in
    ``n_dropped``. p-values are two-sided.
    """
    if method not in ("both", "bootstrap", "permutation"):
        raise ValueError("unknown method {!r}".format(method))
    rates_a, edits_a, len_a = _per_sentence_arrays(result_a)
    rates_b, edits_b, len_b = _per_sentence_arrays(result_b)
    if len(rates_a) != len(rates_b):
        raise ValueError(
            "the two results must cover the same sentences "
            "(got {} and {})".format(len(rates_a), len(rates_b))
        )
    keep = ~np.isnan(rates_a) & ~np.isnan(rates_b)
    n = int(keep.sum())
    dropped = int(len(rates_a) - n)
    if n == 0:
        raise ValueError("no sentence was scored by both systems")

    if statistic == "micro":
        _require_micro(edits_a, len_a)
        _require_micro(edits_b, len_b)
        ea, eb, lengths = edits_a[keep], edits_b[keep], len_a[keep]

        def rate(e, idx):
            return e[idx].sum(axis=-1) / lengths[idx].sum(axis=-1)
    elif statistic == "macro":
        ea, eb, lengths = rates_a[keep], rates_b[keep], None

        def rate(e, idx):
            return e[idx].mean(axis=-1)
    else:
        raise ValueError("unknown statistic {!r}; expected 'macro' or 'micro'".format(statistic))

    whole = np.arange(n)
    rate_a = float(rate(ea, whole))
    rate_b = float(rate(eb, whole))
    delta = rate_a - rate_b

    out = {
        "rate_a": rate_a,
        "rate_b": rate_b,
        "delta": delta,
        "statistic": statistic,
        "n_paired": n,
        "n_dropped": dropped,
        "ci": None,
        "p_bootstrap": None,
        "p_permutation": None,
        "significant": None,
    }

    rng = np.random.default_rng(seed)

    if method in ("both", "bootstrap"):
        deltas = np.empty(n_iterations, dtype=float)
        done = 0
        while done < n_iterations:
            size = min(chunk_size, n_iterations - done)
            idx = rng.integers(0, n, size=(size, n))
            deltas[done:done + size] = rate(ea, idx) - rate(eb, idx)
            done += size
        lo, hi = np.percentile(deltas, percentiles)
        tail = min((deltas <= 0).mean(), (deltas >= 0).mean())
        out["ci"] = (float(lo), float(hi))
        out["p_bootstrap"] = float(min(1.0, 2 * tail))
        out["significant"] = bool(lo > 0 or hi < 0)

    if method in ("both", "permutation"):
        count = 0
        done = 0
        observed = abs(delta)
        while done < n_iterations:
            size = min(chunk_size, n_iterations - done)
            swap = rng.random((size, n)) < 0.5
            perm_a = np.where(swap, eb, ea)
            perm_b = np.where(swap, ea, eb)
            if statistic == "micro":
                permuted = (
                    perm_a.sum(axis=1) / lengths.sum() - perm_b.sum(axis=1) / lengths.sum()
                )
            else:
                permuted = perm_a.mean(axis=1) - perm_b.mean(axis=1)
            count += int((np.abs(permuted) >= observed).sum())
            done += size
        out["p_permutation"] = float((count + 1) / (n_iterations + 1))

    return out

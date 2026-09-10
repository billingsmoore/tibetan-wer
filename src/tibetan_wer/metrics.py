"""Error rates for Tibetan text: CER, SER, and the three SWER variants.

Every metric here is the same Levenshtein rate over a different unit: characters
(CER), tsek-delimited syllables (SER), or words produced by one of the
segmenters in :mod:`tibetan_wer.segmentation` (BoTok-SWER, BERT-SWER,
Gem-SWER). Each returns micro and macro rates, corpus edit-operation totals, and
the per-sentence scores and per-sentence edit counts behind them, which is what
bootstrap intervals (:mod:`tibetan_wer.stats`) and error-structure analysis
(:mod:`tibetan_wer.profile`) need.

Note the argument order throughout: ``(predictions, references)``. Reversing it
leaves substitutions unchanged but swaps insertions and deletions.
"""
from __future__ import annotations

import numpy as np

from .alignment import align, distance, edit_counts, resolve_backend
from .normalization import normalize as _normalize
from .segmentation import (
    TSEK,
    GEMINI_MODEL,
    fallback_counts,
    gemini_segment,
    is_degenerate,
    segment_all,
    syllable_segment,
    word_segment,
)

__all__ = [
    "cer",
    "ser",
    "wer",
    "botok_wer",
    "bert_wer",
    "gemini_wer",
    "score_segments",
    "edit_operations",
    "word_segment",
    "syllable_segment",
]


# ---------------------------------------------------------------------------
# Shared edit-distance engine
# ---------------------------------------------------------------------------

def _wer_from_word_lists(ref_words, hyp_words, backend="auto"):
    """``(S, I, D, score)`` for one pair; ``score`` is None on an empty reference."""
    S, I, D = edit_counts(ref_words, hyp_words, backend=backend)
    r_len = len(ref_words)
    score = (S + I + D) / r_len if r_len > 0 else None
    return S, I, D, score


def _aggregate(pairs, backend: str = "auto", detail: str = "full"):
    """Aggregate ``(ref_units, hyp_units)`` pairs into a result dict.

    A pair may be ``None``, meaning the sentence was not scored (see the
    repetition filter in :func:`gemini_wer`). Such sentences count towards
    ``num_sentences`` and ``num_skipped``, contribute nothing to the totals,
    and appear as ``nan`` in ``per_utterance``.

    ``detail='rates'`` computes distances only. The rates are identical to
    ``'full'``; what it gives up is the split into substitutions, insertions
    and deletions, and in exchange it can use the fast backend, which is around
    a thousand times quicker.
    """
    if detail not in ("full", "rates"):
        raise ValueError(f"unknown detail {detail!r}; expected 'full' or 'rates'")
    want_ops = detail == "full"
    backend = resolve_backend(backend, need_operations=want_ops)

    total_S = total_I = total_D = total_edits = total_ref = 0
    per_utt = []
    per_sentence = []
    scored = []
    skipped = 0
    for pair in pairs:
        if pair is None:
            skipped += 1
            scored.append(float("nan"))
            per_sentence.append(None)
            continue
        ref_words, hyp_words = pair
        r_len = len(ref_words)
        if want_ops:
            S, I, D, score = _wer_from_word_lists(ref_words, hyp_words, backend=backend)
            edits = S + I + D
            total_S += S
            total_I += I
            total_D += D
            sentence = {
                "substitutions": S,
                "insertions": I,
                "deletions": D,
                "edits": edits,
                "reference_length": r_len,
                "score": score,
            }
        else:
            edits = distance(ref_words, hyp_words, backend=backend)
            score = edits / r_len if r_len > 0 else None
            sentence = {
                "substitutions": None,
                "insertions": None,
                "deletions": None,
                "edits": edits,
                "reference_length": r_len,
                "score": score,
            }
        total_edits += edits
        total_ref += r_len
        per_sentence.append(sentence)
        scored.append(score if score is not None else float("nan"))
        if score is not None:
            per_utt.append(score)
    micro = total_edits / total_ref if total_ref > 0 else float("nan")
    macro = float(np.mean(per_utt)) if per_utt else float("nan")
    return {
        "micro_wer": float(micro),
        "macro_wer": float(macro),
        "substitutions": total_S if want_ops else None,
        "insertions": total_I if want_ops else None,
        "deletions": total_D if want_ops else None,
        "num_sentences": len(per_sentence),
        "num_scored": len(per_utt),
        "num_skipped": skipped,
        "per_utterance": np.array(scored, dtype=float),
        "per_sentence": per_sentence,
    }


def _relabel(result: dict, unit: str) -> dict:
    """Rename ``micro_wer``/``macro_wer`` for the metrics that are not WER."""
    if unit == "wer":
        return result
    out = dict(result)
    out["micro_" + unit] = out.pop("micro_wer")
    out["macro_" + unit] = out.pop("macro_wer")
    return out


def _validate_inputs(predictions, references):
    """Normalize str->list and check lengths match. Returns (predictions, references)."""
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    predictions = list(predictions)
    references = list(references)
    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must have the same length "
            "(got {} and {})".format(len(predictions), len(references))
        )
    return predictions, references


def _apply_normalization(texts, normalize):
    """``normalize`` may be False, True, or a dict of :func:`normalize` kwargs."""
    if not normalize:
        return texts
    kwargs = normalize if isinstance(normalize, dict) else {}
    return [_normalize(t, **kwargs) for t in texts]


def _with_fallbacks(result: dict, before: dict) -> dict:
    """Attach the number of segmentation fallbacks incurred by this call."""
    after = fallback_counts()
    result["num_segmentation_fallbacks"] = sum(after.values()) - sum(before.values())
    return result


# ---------------------------------------------------------------------------
# Metrics over units the caller supplies
# ---------------------------------------------------------------------------

def score_segments(
    prediction_segments,
    reference_segments,
    unit: str = "wer",
    backend: str = "auto",
    detail: str = "full",
) -> dict:
    """Score already-segmented text.

    Parameters
    ----------
    prediction_segments, reference_segments : sequence of sequence of str
        Token lists, e.g. from :func:`tibetan_wer.segmentation.segment_all`.
        An entry may be ``None`` to mark a sentence that could not be
        segmented; it is reported as ``nan`` rather than scored.
    unit : str
        Label for the rate keys: ``"wer"`` gives ``micro_wer``/``macro_wer``,
        ``"cer"`` gives ``micro_cer``/``macro_cer``, and so on.
    backend, detail
        See :func:`cer`.

    Notes
    -----
    Use this to segment once and score many systems against the same
    segmentation -- worthwhile for the BERT and Gemini segmenters, where
    segmentation dominates the cost of an evaluation.
    """
    preds, refs = _validate_inputs(prediction_segments, reference_segments)
    pairs = [
        None if (ref is None or pred is None) else (list(ref), list(pred))
        for pred, ref in zip(preds, refs)
    ]
    return _relabel(_aggregate(pairs, backend=backend, detail=detail), unit)


def edit_operations(prediction, reference, level: str = "char", backend: str = "auto", **segment_kwargs):
    """The individual edits between one prediction and one reference.

    Parameters
    ----------
    prediction, reference : str
    level : {'char', 'syllable', 'botok', 'bert', 'gemini'}
        Unit to align over. ``'char'`` is the CER unit, ``'syllable'`` the SER
        unit, the rest the SWER units.
    backend : {'auto', 'python', 'fast'}
        See :func:`cer`. ``'auto'`` keeps the reference alignment here, since
        which operations are reported is exactly what the backends can differ
        on.
    **segment_kwargs
        Forwarded to the segmenter.

    Returns
    -------
    list of :class:`tibetan_wer.alignment.EditOperation`
        Each carries the operation (``"S"``, ``"I"`` or ``"D"``), the reference
        and hypothesis units involved, and their positions -- enough to build
        confusion tables, count edits falling on the tsek, or classify errors
        by the character they land on.
    """
    if level == "char":
        ref_units, hyp_units = list(reference), list(prediction)
    elif level == "syllable":
        ref_units, hyp_units = syllable_segment(reference), syllable_segment(prediction)
    else:
        ref_units, hyp_units = segment_all(
            [reference, prediction], method=level, **segment_kwargs
        )
    return align(ref_units, hyp_units, backend=backend)


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------

def cer(predictions, references, normalize=False, backend: str = "auto", detail: str = "full") -> dict:
    """Character Error Rate.

    Parameters
    ----------
    predictions, references : str or sequence of str
    normalize : bool or dict
        ``False`` (default) scores text as stored: no Unicode normalization, no
        case folding, no punctuation stripping. ``True`` applies
        :func:`tibetan_wer.normalization.normalize`; a dict passes keyword
        arguments to it. CER is sensitive to these conventions, so state which
        you used when reporting.
    backend : {'auto', 'python', 'fast'}
        Alignment implementation. ``'fast'`` needs ``rapidfuzz`` or
        ``python-Levenshtein`` and returns identical rates but may split a
        total differently between substitutions, insertions and deletions, so
        ``'auto'`` uses it only where operation counts are not requested.
    detail : {'full', 'rates'}
        ``'rates'`` skips the operation counts, which lets the fast backend run
        -- around a thousand times quicker on long lines, with identical rates.

    Returns
    -------
    dict with ``micro_cer``, ``macro_cer``, ``substitutions``, ``insertions``,
    ``deletions``, ``num_sentences``, ``num_scored``, ``num_skipped``,
    ``per_utterance``, ``per_sentence``.
    """
    predictions, references = _validate_inputs(predictions, references)
    predictions = _apply_normalization(predictions, normalize)
    references = _apply_normalization(references, normalize)
    pairs = [(list(ref), list(pred)) for pred, ref in zip(predictions, references)]
    return _relabel(_aggregate(pairs, backend=backend, detail=detail), "cer")


# ---------------------------------------------------------------------------
# SER (tsek-split syllable error rate)
# ---------------------------------------------------------------------------

def ser(predictions, references, normalize=False, backend: str = "auto", detail: str = "full") -> dict:
    """Syllable Error Rate, using the tsek (་) as the syllable boundary.

    Parameters as :func:`cer`.

    Returns
    -------
    dict with ``micro_ser``, ``macro_ser``, and the shared keys documented in
    :func:`cer`.
    """
    predictions, references = _validate_inputs(predictions, references)
    predictions = _apply_normalization(predictions, normalize)
    references = _apply_normalization(references, normalize)
    pairs = [
        (syllable_segment(ref), syllable_segment(pred))
        for pred, ref in zip(predictions, references)
    ]
    return _relabel(_aggregate(pairs, backend=backend, detail=detail), "ser")


# ---------------------------------------------------------------------------
# SWER variants
# ---------------------------------------------------------------------------

def wer(
    predictions,
    references,
    normalize=False,
    cache: dict | None = None,
    backend: str = "auto",
    detail: str = "full",
    on_error: str = "fallback",
) -> dict:
    """BoTok-SWER: WER after botok word segmentation.

    ``cache`` is an optional ``{text: tokens}`` map, reused and updated in place
    across calls. ``on_error='raise'`` propagates a segmenter failure instead of
    falling back to a syllable split for that string; either way the result
    carries ``num_segmentation_fallbacks``.

    Other parameters as :func:`cer`.
    """
    predictions, references = _validate_inputs(predictions, references)
    predictions = _apply_normalization(predictions, normalize)
    references = _apply_normalization(references, normalize)
    before = fallback_counts()
    segments = segment_all(
        list(references) + list(predictions), method="botok", cache=cache, on_error=on_error
    )
    n = len(references)
    result = score_segments(segments[n:], segments[:n], unit="wer", backend=backend, detail=detail)
    return _with_fallbacks(result, before)


botok_wer = wer


def bert_wer(
    predictions,
    references,
    device=None,
    normalize=False,
    cache: dict | None = None,
    backend: str = "auto",
    detail: str = "full",
    max_syllables: int | None = None,
    on_error: str = "fallback",
) -> dict:
    """BERT-SWER: WER after tibetan-bert-base-upos segmentation.

    ``device`` is passed to the transformers pipeline; ``None`` auto-detects
    CUDA. Distinct strings are segmented once each. ``max_syllables`` windows
    long lines to stay inside the model's 512-token limit -- see
    :func:`tibetan_wer.segmentation.bert_segment`.

    Other parameters as :func:`wer`.
    """
    predictions, references = _validate_inputs(predictions, references)
    predictions = _apply_normalization(predictions, normalize)
    references = _apply_normalization(references, normalize)
    before = fallback_counts()
    segments = segment_all(
        list(references) + list(predictions),
        method="bert",
        cache=cache,
        device=device,
        max_syllables=max_syllables,
        on_error=on_error,
    )
    n = len(references)
    result = score_segments(segments[n:], segments[:n], unit="wer", backend=backend, detail=detail)
    return _with_fallbacks(result, before)


def gemini_wer(
    predictions,
    references,
    api_key=None,
    model: str = GEMINI_MODEL,
    max_repetition_ratio: float | None = 10.0,
    max_output_tokens: int | None = None,
    normalize=False,
    cache: dict | None = None,
    backend: str = "auto",
    detail: str = "full",
    workers: int = 1,
    on_error: str = "raise",
) -> dict:
    """Gem-SWER: WER after Gemini word segmentation.

    Parameters
    ----------
    predictions, references : list of str or str
    api_key : str or None
        Defaults to the ``GEMINI_API_KEY`` environment variable.
    model : str
    max_repetition_ratio : float or None
        Sentences whose prediction or reference repeats characters more than
        this many times over (length / distinct characters) are not sent to the
        API and are reported as ``nan``, since such strings -- the output of a
        collapsed model -- provoke request timeouts. ``None`` disables the
        filter and restores pre-1.2 behaviour. Check ``num_skipped``: a rate
        computed over the remainder is not comparable with one computed over
        all sentences, and a system is degraded, not undefined, on the
        sentences that were dropped.
    max_output_tokens : int or None
        Caps each response, bounding the cost of a repetition loop.
    cache : dict or None
        ``{text: tokens}`` map, reused and updated in place. Persist it to
        avoid paying for the same segmentation twice.
    workers : int
        Concurrent API calls. The work is network latency, so this scales
        nearly linearly; the per-call pause is dropped above 1.
    on_error : {'raise', 'fallback'}
        What to do with a string the API will not segment after its retries.

    Other parameters as :func:`cer`.

    Returns
    -------
    dict with the keys documented in :func:`cer`, plus ``num_skipped`` and
    ``num_segmentation_fallbacks``.
    """
    predictions, references = _validate_inputs(predictions, references)
    predictions = _apply_normalization(predictions, normalize)
    references = _apply_normalization(references, normalize)
    threshold = max_repetition_ratio
    skip = [
        threshold is not None
        and (is_degenerate(ref, threshold) or is_degenerate(pred, threshold))
        for pred, ref in zip(predictions, references)
    ]
    cache = {} if cache is None else cache
    before = fallback_counts()
    wanted = [
        text
        for keep, pred, ref in zip(skip, predictions, references)
        if not keep
        for text in (pred, ref)
    ]
    segment_all(
        wanted,
        method="gemini",
        cache=cache,
        workers=workers,
        api_key=api_key,
        model=model,
        max_output_tokens=max_output_tokens,
        on_error=on_error,
    )
    pred_segments = [None if s else cache[p] for s, p in zip(skip, predictions)]
    ref_segments = [None if s else cache[r] for s, r in zip(skip, references)]
    result = score_segments(
        pred_segments, ref_segments, unit="wer", backend=backend, detail=detail
    )
    return _with_fallbacks(result, before)

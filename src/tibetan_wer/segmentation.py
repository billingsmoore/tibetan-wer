"""Segmenters, exposed on their own.

Every metric in this package is a Levenshtein rate over some unit sequence, and
the only thing that differs between SER, BoTok-SWER, BERT-SWER and Gem-SWER is
how a string is cut into units. Those cuts are available here directly, so a
caller can segment once, cache the result, and score many systems against it
(the neural and API segmenters are far more expensive than the alignment), or
inspect the segmentation itself.

The segmenters are the ones used in the SWER paper: the same botok tokenizer,
the same UPOS model, the same Gemini model, prompt and temperature.

**Fallbacks are counted.** A segmenter that fails on a string falls back to a
syllable split rather than losing the run, which silently makes that line's
"word" rate a syllable rate. Every fallback is tallied in
:func:`fallback_counts`, and the metrics report the tally as
``num_segmentation_fallbacks``; treat a nonzero count as a caveat on the
number, not as noise.
"""
from __future__ import annotations

import collections
import os
import time
from typing import Iterable

__all__ = [
    "TSEK",
    "GEMINI_MODEL",
    "BERT_UPOS_MODEL",
    "GEMINI_SEGMENT_PROMPT",
    "syllable_segment",
    "word_segment",
    "botok_segment",
    "bert_segment",
    "gemini_segment",
    "segment_all",
    "repetition_ratio",
    "is_degenerate",
    "fallback_counts",
    "reset_fallback_counts",
]

TSEK = "་"  # ་ intersyllabic tsek

GEMINI_MODEL = "gemini-2.5-flash-lite"
BERT_UPOS_MODEL = "KoichiYasuoka/tibetan-bert-base-upos"

GEMINI_SEGMENT_PROMPT = (
    "Segment the following Tibetan text into words. "
    "Output ONLY the segmented text with words separated by a pipe character (|). "
    "Keep the original Tibetan characters and tsek marks (་) intact within each word. "
    "Do not add any explanation or punctuation beyond the pipe separators.\n\n"
    "Text: {text}"
)

_botok_tokenizer = None
_bert_nlp = None
_gemini_client = None

_fallbacks: collections.Counter = collections.Counter()


# ---------------------------------------------------------------------------
# Fallback accounting
# ---------------------------------------------------------------------------

def fallback_counts() -> dict:
    """``{segmenter: n}`` strings that fell back to a syllable split so far."""
    return dict(_fallbacks)


def reset_fallback_counts() -> dict:
    """Zero the tally and return what it held."""
    previous = dict(_fallbacks)
    _fallbacks.clear()
    return previous


def _fallback(method: str, text: str, error: Exception, on_error: str) -> list:
    if on_error == "raise":
        raise error
    if on_error not in ("fallback", "raise"):
        raise ValueError(f"unknown on_error {on_error!r}; expected 'fallback' or 'raise'")
    _fallbacks[method] += 1
    return syllable_segment(text)


# ---------------------------------------------------------------------------
# Degenerate-output detection
# ---------------------------------------------------------------------------

def repetition_ratio(text: str) -> float:
    """Length divided by number of distinct characters.

    A collapsed recognizer emits long runs of a handful of characters, which
    drives this ratio up without bound; ordinary Tibetan lines sit well below
    10. Used to keep such strings away from the API segmenter, where they cause
    request timeouts rather than merely bad segmentations.
    """
    if not text:
        return 0.0
    return len(text) / max(len(set(text)), 1)


def is_degenerate(text: str, threshold: float = 10.0) -> bool:
    """True if ``text`` repeats characters more than ``threshold`` times over."""
    return repetition_ratio(text) > threshold


# ---------------------------------------------------------------------------
# Syllables
# ---------------------------------------------------------------------------

def syllable_segment(text: str) -> list:
    """Split on the tsek (U+0F0B), discarding empty segments.

    Tsek bstar (U+0F0C) is deliberately not treated as a boundary; fold it with
    :func:`tibetan_wer.normalization.normalize` first if you want it to be.
    """
    return [s for s in text.split(TSEK) if s]


# ---------------------------------------------------------------------------
# botok
# ---------------------------------------------------------------------------

def _get_botok_tokenizer():
    global _botok_tokenizer
    if _botok_tokenizer is None:
        import botok
        _botok_tokenizer = botok.WordTokenizer()
    return _botok_tokenizer


def word_segment(sentence: str, tokenizer=None, on_error: str = "fallback") -> list:
    """Segment a Tibetan sentence into words with botok.

    Parameters
    ----------
    sentence : str
    tokenizer : optional
        A botok-compatible tokenizer. Defaults to the shared lazy instance.
    on_error : {'fallback', 'raise'}
        botok's merge steps can raise on pathological input -- a degenerate,
        highly repetitive hypothesis from a collapsed model. ``'fallback'``
        substitutes a syllable split for that string and counts it in
        :func:`fallback_counts`; ``'raise'`` propagates, as in 1.1.x.

    Returns
    -------
    list of str

    Notes
    -----
    Tokens are returned exactly as botok emits them, empty ones included, so
    that values match those published with the SWER paper.
    """
    tok = tokenizer if tokenizer is not None else _get_botok_tokenizer()
    try:
        tokens = tok.tokenize(sentence.strip())
        return [t["text_cleaned"] for t in tokens]
    except Exception as exc:  # noqa: BLE001
        return _fallback("botok", sentence, exc, on_error)


botok_segment = word_segment


# ---------------------------------------------------------------------------
# BERT-UPOS
# ---------------------------------------------------------------------------

def _get_bert_nlp(device=None):
    global _bert_nlp
    if _bert_nlp is None:
        import torch
        from transformers import pipeline as hf_pipeline
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        _bert_nlp = hf_pipeline(
            "token-classification",
            BERT_UPOS_MODEL,
            trust_remote_code=True,
            aggregation_strategy="simple",
            device=device,
        )
    return _bert_nlp


def bert_segment(
    text: str,
    device=None,
    nlp=None,
    max_syllables: int | None = None,
    on_error: str = "fallback",
) -> list:
    """Segment with the tibetan-bert-base-upos token classifier.

    Parameters
    ----------
    max_syllables : int or None
        The model reads at most 512 tokens. A longer line either raises or is
        silently truncated, and a truncated segmentation scores as a stream of
        deletions that belong to the segmenter rather than to the system under
        test. Setting this splits the text into windows of at most that many
        syllables, segments each, and concatenates -- boundaries at the window
        edges are the cost. Lines in the SWER paper's data sit well inside the
        limit, so the default leaves the behaviour unchanged; set it when
        scoring long lines.
    on_error : {'fallback', 'raise'}
        As :func:`word_segment`.
    """
    if not text or not text.strip():
        return []
    pipeline = nlp if nlp is not None else _get_bert_nlp(device)
    if max_syllables:
        syllables = syllable_segment(text)
        if len(syllables) > max_syllables:
            out = []
            for start in range(0, len(syllables), max_syllables):
                window = TSEK.join(syllables[start:start + max_syllables]) + TSEK
                out.extend(bert_segment(window, nlp=pipeline, on_error=on_error))
            return out
    try:
        result = pipeline(text)
        words = [e["word"].strip() for e in result if e["word"].strip()]
        if words:
            return words
        _fallbacks["bert"] += 1
        return syllable_segment(text)
    except Exception as exc:  # noqa: BLE001
        return _fallback("bert", text, exc, on_error)


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _get_gemini_client(api_key=None):
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key required: pass api_key= or set GEMINI_API_KEY")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def gemini_segment(
    text: str,
    api_key: str | None = None,
    model: str = GEMINI_MODEL,
    max_retries: int = 3,
    max_output_tokens: int | None = None,
    client=None,
    pause: float = 0.2,
    on_error: str = "raise",
) -> list:
    """Segment with Gemini at temperature 0.

    ``max_output_tokens`` caps the response, bounding the cost of a repetition
    loop. ``pause`` is the delay after each call; set it to 0 when calling
    concurrently and rely on the retry instead. ``on_error='fallback'`` returns
    a syllable split for a string the API will not segment, counted in
    :func:`fallback_counts`.
    """
    cli = client if client is not None else _get_gemini_client(api_key)
    prompt = GEMINI_SEGMENT_PROMPT.format(text=text)
    config = {"temperature": 0.0}
    if max_output_tokens is not None:
        config["max_output_tokens"] = max_output_tokens
    raw = ""
    for attempt in range(max_retries):
        try:
            response = cli.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            raw = (response.text or "").strip()
            break
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return _fallback("gemini", text, exc, on_error)
    if pause:
        time.sleep(pause)
    if "|" in raw:
        return [w.strip() for w in raw.split("|") if w.strip()]
    return raw.strip().split()


# ---------------------------------------------------------------------------
# Batch segmentation with de-duplication
# ---------------------------------------------------------------------------

_METHODS = {
    "syllable": syllable_segment,
    "botok": word_segment,
    "bert": bert_segment,
    "gemini": gemini_segment,
}


def segment_all(
    texts: Iterable[str],
    method: str = "botok",
    cache: dict | None = None,
    workers: int = 1,
    **kwargs,
) -> list:
    """Segment many strings, calling the segmenter once per distinct string.

    Parameters
    ----------
    texts : iterable of str
    method : {'syllable', 'botok', 'bert', 'gemini'}
    cache : dict, optional
        Mutable ``{text: tokens}`` map, read and updated in place. Pass one to
        reuse segmentations across calls -- across the systems being compared,
        or across sessions if persisted.
    workers : int
        Threads for the API segmenter, where the wait is network latency. Above
        1 this also drops ``gemini_segment``'s default pause, since the
        concurrency limit takes over from the delay. Ignored by the local
        segmenters, which are CPU- or GPU-bound.
    **kwargs
        Forwarded to the segmenter (``device=``, ``max_syllables=`` for bert;
        ``api_key=``, ``model=``, ``max_output_tokens=`` for gemini;
        ``on_error=`` for all three).

    Returns
    -------
    list of list of str, aligned with ``texts``.
    """
    if method not in _METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {sorted(_METHODS)}")
    fn = _METHODS[method]
    texts = list(texts)
    cache = {} if cache is None else cache
    if method == "bert":
        kwargs.setdefault("nlp", _get_bert_nlp(kwargs.pop("device", None)))
    pending = []
    seen = set()
    for text in texts:
        if text not in cache and text not in seen:
            seen.add(text)
            pending.append(text)
    if workers > 1 and pending:
        if method == "gemini":
            kwargs.setdefault("client", _get_gemini_client(kwargs.get("api_key")))
            kwargs.setdefault("pause", 0.0)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for text, tokens in zip(pending, pool.map(lambda t: fn(t, **kwargs), pending)):
                cache[text] = tokens
    else:
        for text in pending:
            cache[text] = fn(text, **kwargs)
    return [cache[t] for t in texts]

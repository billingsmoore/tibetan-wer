import os
import time
import numpy as np

TSEK = "་"  # U+0F0B


# ---------------------------------------------------------------------------
# Shared edit-distance WER engine
# ---------------------------------------------------------------------------

def _wer_from_word_lists(ref_words, hyp_words):
    r_len = len(ref_words)
    p_len = len(hyp_words)

    d = np.zeros((r_len + 1, p_len + 1), dtype=np.int32)
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(p_len + 1):
        d[0][j] = j
    for i in range(1, r_len + 1):
        for j in range(1, p_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )

    i, j = r_len, p_len
    S = I = D = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            S += 1; i -= 1; j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            I += 1; j -= 1
        else:
            D += 1; i -= 1

    score = (S + I + D) / r_len if r_len > 0 else None
    return S, I, D, score


def _aggregate(pairs):
    total_S = total_I = total_D = total_ref = 0
    per_utt = []
    for ref_words, hyp_words in pairs:
        S, I, D, score = _wer_from_word_lists(ref_words, hyp_words)
        total_S += S
        total_I += I
        total_D += D
        total_ref += len(ref_words)
        if score is not None:
            per_utt.append(score)
    micro = (total_S + total_I + total_D) / total_ref if total_ref > 0 else float("inf")
    macro = float(np.mean(per_utt)) if per_utt else float("inf")
    return {
        "micro_wer": float(micro),
        "macro_wer": float(macro),
        "substitutions": total_S,
        "insertions": total_I,
        "deletions": total_D,
        "num_sentences": len(pairs),
    }


def _validate_inputs(predictions, references):
    """Normalize str->list and check lengths match. Returns (predictions, references)."""
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions and references must have the same length "
            f"(got {len(predictions)} and {len(references)})"
        )
    return predictions, references


# ---------------------------------------------------------------------------
# Botok segmenter
# ---------------------------------------------------------------------------

_botok_tokenizer = None


def _get_botok_tokenizer():
    global _botok_tokenizer
    if _botok_tokenizer is None:
        import botok
        _botok_tokenizer = botok.WordTokenizer()
    return _botok_tokenizer


def _botok_segment(text):
    tok = _get_botok_tokenizer()
    tokens = tok.tokenize(text.strip())
    return [t["text_cleaned"] for t in tokens]


def word_segment(sentence, tokenizer=None):
    """Segment a Tibetan sentence into words using botok.

    Parameters
    ----------
    sentence : str
    tokenizer : optional
        A botok-compatible tokenizer. Defaults to the shared lazy instance.

    Returns
    -------
    list of str
    """
    if tokenizer is not None:
        tokens = tokenizer.tokenize(sentence.strip())
        return [t["text_cleaned"] for t in tokens]
    return _botok_segment(sentence)


def wer(predictions, references):
    """WER using botok word segmentation.

    Parameters
    ----------
    predictions : list of str or str
    references  : list of str or str

    Returns
    -------
    dict with micro_wer, macro_wer, substitutions, insertions, deletions,
    num_sentences.
    """
    predictions, references = _validate_inputs(predictions, references)
    pairs = [
        (_botok_segment(ref), _botok_segment(pred))
        for pred, ref in zip(predictions, references)
    ]
    return _aggregate(pairs)


botok_wer = wer


# ---------------------------------------------------------------------------
# SER (tsek-split syllable error rate)
# ---------------------------------------------------------------------------

def ser(predictions, references):
    """Syllable Error Rate using tsek (་) as the syllable boundary.

    Parameters
    ----------
    predictions : list of str or str
    references  : list of str or str

    Returns
    -------
    dict with micro_ser, macro_ser, substitutions, insertions, deletions,
    num_sentences.
    """
    predictions, references = _validate_inputs(predictions, references)
    pairs = [
        ([s for s in ref.split(TSEK) if s], [s for s in pred.split(TSEK) if s])
        for pred, ref in zip(predictions, references)
    ]
    result = _aggregate(pairs)
    return {
        "micro_ser": result["micro_wer"],
        "macro_ser": result["macro_wer"],
        "substitutions": result["substitutions"],
        "insertions": result["insertions"],
        "deletions": result["deletions"],
        "num_sentences": result["num_sentences"],
    }


# ---------------------------------------------------------------------------
# BERT-UPOS segmenter
# ---------------------------------------------------------------------------

_bert_nlp = None


def _get_bert_nlp(device=None):
    global _bert_nlp
    if _bert_nlp is None:
        from transformers import pipeline as hf_pipeline
        import torch
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        _bert_nlp = hf_pipeline(
            "token-classification",
            "KoichiYasuoka/tibetan-bert-base-upos",
            trust_remote_code=True,
            aggregation_strategy="simple",
            device=device,
        )
    return _bert_nlp


def _bert_segment(text, nlp):
    if not text or not text.strip():
        return []
    try:
        result = nlp(text)
        words = [e["word"].strip() for e in result if e["word"].strip()]
        return words if words else [s for s in text.split(TSEK) if s]
    except Exception:
        return [s for s in text.split(TSEK) if s]


def bert_wer(predictions, references, device=None):
    """WER using KoichiYasuoka/tibetan-bert-base-upos segmentation.

    Parameters
    ----------
    predictions : list of str or str
    references  : list of str or str
    device      : int or None
        Passed to the transformers pipeline. None auto-detects CUDA.

    Returns
    -------
    dict with micro_wer, macro_wer, substitutions, insertions, deletions,
    num_sentences.
    """
    predictions, references = _validate_inputs(predictions, references)
    nlp = _get_bert_nlp(device)
    unique = set(predictions) | set(references)
    cache = {text: _bert_segment(text, nlp) for text in unique}
    pairs = [
        (cache[ref], cache[pred])
        for pred, ref in zip(predictions, references)
    ]
    return _aggregate(pairs)


# ---------------------------------------------------------------------------
# Gemini segmenter
# ---------------------------------------------------------------------------

def _gemini_segment(client, text, model, max_retries=3):
    prompt = (
        "Segment the following Tibetan text into words. "
        "Output ONLY the segmented text with words separated by a pipe character (|). "
        "Keep the original Tibetan characters and tsek marks (་) intact within each word. "
        "Do not add any explanation or punctuation beyond the pipe separators.\n\n"
        f"Text: {text}"
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config={"temperature": 0.0},
            )
            return response.text.strip()
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def _parse_gemini_words(raw):
    if "|" in raw:
        return [w.strip() for w in raw.split("|") if w.strip()]
    return raw.strip().split()


def gemini_wer(predictions, references, api_key=None, model="gemini-2.5-flash-lite"):
    """WER using Gemini word segmentation.

    Parameters
    ----------
    predictions : list of str or str
    references  : list of str or str
    api_key     : str or None
        Gemini API key. Defaults to the GEMINI_API_KEY environment variable.
    model       : str
        Gemini model name.

    Returns
    -------
    dict with micro_wer, macro_wer, substitutions, insertions, deletions,
    num_sentences.
    """
    predictions, references = _validate_inputs(predictions, references)
    from google import genai

    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key required: pass api_key= or set GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)
    unique = set(predictions) | set(references)
    cache = {}
    for text in unique:
        raw = _gemini_segment(client, text, model)
        cache[text] = _parse_gemini_words(raw)
        time.sleep(0.2)

    pairs = [
        (cache[ref], cache[pred])
        for pred, ref in zip(predictions, references)
    ]
    return _aggregate(pairs)

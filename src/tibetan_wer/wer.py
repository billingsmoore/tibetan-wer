import botok
import numpy as np

tokenizer = botok.WordTokenizer()

def word_segment(sentence, tokenizer=tokenizer):
    """
    Segment a sentence into words using a custom tokenizer.

    Parameters
    ----------
    sentence : str
        The input sentence to be segmented.
    tokenizer : callable, optional
        A tokenizer function or object with a `.tokenize()` method.
        Default is the globally defined `tokenizer`.

    Returns
    -------
    list of str
        A list of cleaned word strings extracted from the input sentence.
    """

    tokens = tokenizer.tokenize(sentence.strip())
    
    words = [elt['text_cleaned'] for elt in tokens]

    return words

def wer(predictions, references):
    """
    Compute micro- and macro-averaged Word Error Rate (WER) over a batch of sentences.

    Parameters
    ----------
    predictions : list of str
        List of predicted sentences from the ASR model.
    references : list of str
        List of ground truth sentences.

    Returns
    -------
    result : dict
        A dictionary with:
            - 'micro_wer': WER over the whole corpus (total errors / total reference words)
            - 'macro_wer': Mean WER over individual examples
            - 'substitutions': Total substitutions
            - 'insertions': Total insertions
            - 'deletions': Total deletions
            - 'num_sentences': Number of sentence pairs
    """
    total_S = total_I = total_D = total_ref_words = 0
    wer_scores = []

    for pred, ref in zip(predictions, references):
        pred_words = word_segment(pred)
        ref_words = word_segment(ref)

        p_len = len(pred_words)
        r_len = len(ref_words)

        # DP matrix
        d = np.zeros((r_len + 1, p_len + 1), dtype=np.int32)
        for i in range(r_len + 1):
            d[i][0] = i
        for j in range(p_len + 1):
            d[0][j] = j

        for i in range(1, r_len + 1):
            for j in range(1, p_len + 1):
                if ref_words[i - 1] == pred_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(
                        d[i - 1][j] + 1,     # deletion
                        d[i][j - 1] + 1,     # insertion
                        d[i - 1][j - 1] + 1  # substitution
                    )

        # Backtrace for individual errors
        i, j = r_len, p_len
        S = I = D = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == pred_words[j - 1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
                S += 1
                i -= 1
                j -= 1
            elif j > 0 and d[i][j] == d[i][j - 1] + 1:
                I += 1
                j -= 1
            elif i > 0 and d[i][j] == d[i - 1][j] + 1:
                D += 1
                i -= 1

        total_S += S
        total_I += I
        total_D += D
        total_ref_words += r_len
        if r_len > 0:
            wer_scores.append((S + I + D) / r_len)

    micro_wer = (total_S + total_I + total_D) / total_ref_words if total_ref_words > 0 else float("inf")
    macro_wer = np.mean(wer_scores) if wer_scores else float("inf")

    return {
        "micro_wer": micro_wer,
        "macro_wer": macro_wer,
        "substitutions": total_S,
        "insertions": total_I,
        "deletions": total_D,
        "num_sentences": len(predictions)
    }
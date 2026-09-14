> **Status: done.** The three items below -- CER, per-utterance exposure and
> bootstrap CIs -- ship in 1.2.0, along with a good deal more; see CHANGELOG.md.
> The four listed under *not brought over* stay out: FOM and CDD as too niche,
> and `correlation_matrix` / `pool_per_utterance` because with `per_utterance`
> exposed they are a two-line call on the caller's side. This file is kept as
> the record of where the additions came from.

# Potential enhancements from the OCR metric study

These are gaps and additions identified while building the evaluation pipeline for
"When Character Error Rate Fails to Predict Word Error: An Empirical Study of OCR
Metrics Across Tibetan Script Domains" (companion to the ASR paper). All of the
code below lives in `src/metrics.py` in the OCR paper repo.

---

## 1. CER — Character Error Rate

The package has no `cer()` function. It's the most common OCR metric in the
literature and the natural baseline against which SER/WER are compared.

```python
def cer(predictions, references):
    predictions, references = _validate_inputs(predictions, references)
    pairs = [(list(ref), list(pred)) for pred, ref in zip(predictions, references)]
    result = _aggregate(pairs)
    return {
        "micro_cer": result["micro_wer"],
        "macro_cer": result["macro_wer"],
        "substitutions": result["substitutions"],
        "insertions": result["insertions"],
        "deletions": result["deletions"],
        "num_sentences": result["num_sentences"],
    }
```

---

## 2. Per-utterance scores in return values

`_aggregate` already builds `per_utt` internally but throws it away. Exposing it
unlocks bootstrap CI, correlation analysis, and model-ranking comparisons without
requiring callers to reimplement the alignment logic.

Backward-compatible change: add `"per_utterance"` to the returned dict (a numpy
array of per-sentence scores, `nan` where the reference was empty).

```python
def _aggregate(pairs):
    total_S = total_I = total_D = total_ref = 0
    per_utt = []
    scored = []
    for ref_words, hyp_words in pairs:
        S, I, D, score = _wer_from_word_lists(ref_words, hyp_words)
        total_S += S; total_I += I; total_D += D
        total_ref += len(ref_words)
        scored.append(score if score is not None else float("nan"))
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
        "per_utterance": np.array(scored),   # <-- new
    }
```

---

## 3. Bootstrap confidence intervals

With per-utterance scores exposed (above), a `bootstrap_ci` helper is a natural
addition. This is what the OCR and ASR papers use for the significance claims:
macro-averaged bootstrap at 10k iterations.

```python
def bootstrap_ci(per_utterance, n_iterations=10_000, percentiles=(2.5, 97.5), seed=42):
    """Return (mean, lo, hi) 95% CI over macro-averaged utterance scores."""
    values = per_utterance[~np.isnan(per_utterance)]
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.array([rng.choice(values, size=n, replace=True).mean()
                      for _ in range(n_iterations)])
    lo, hi = np.percentile(means, percentiles)
    return float(values.mean()), float(lo), float(hi)
```

Usage:

```python
result = ser(predictions, references)
mean, lo, hi = bootstrap_ci(result["per_utterance"])
# e.g. SER = 0.142 [0.138, 0.146]
```

---

## Notes on what was *not* brought over

- **FOM (Figure of Merit, Kluzner 2009)** — substitution-weighted error metric
  (subs penalized 5×). Used in the study but too niche for a general package.
- **CDD (Character Distribution Divergence, Bourne 2026)** — Jensen-Shannon
  divergence over character frequency distributions. Same: niche and depends on
  scipy.
- **`correlation_matrix` / `pool_per_utterance`** — research analysis tools for
  the correlation study specifically. Not general-purpose enough.

The three items above (CER, per-utterance exposure, bootstrap CI) are the ones
that would be useful to any practitioner evaluating a Tibetan OCR or ASR system,
independent of the study.

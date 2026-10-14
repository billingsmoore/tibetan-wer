# Changelog

## 2.0.0

**SER values change.** Syllables were split on the tsek (U+0F0B) alone. A syllable that ends a clause carries a shad and no tsek, and one before a space carries neither, so a tsek-only split fused it with the next syllable: `པས། ལུགས`, `གནད།  གངས`, `ཛཿ  ཀི` were each one unit, failed whole if either syllable was wrong, and carried the shad and spaces as part of the unit text. On OCR test references from four Tibetan script domains, 2–8% of tsek-split units spanned two syllables.

The major version marks the change in what `ser` returns, not an API change: every signature is as in 1.2.0. CER and BoTok-, BERT- and Gem-SWER are unchanged.

### Changed

- `syllable_segment`, and so `ser`, `edit_operations(level="syllable")`, `error_profile(level="syllable")`, `segment_all(method="syllable")`, the CLI `--vocabulary` set and the syllable fallback of the word segmenters, now split on whitespace, tsek, tsek bstar, the shads and the other syllable-closing marks, and drop them. The full rule is in the README under *Syllables*; the delimiter class is exported as `SYLLABLE_DELIMITERS`.
- Punctuation is no longer scored by SER. A missing or extra shad cost a syllable error, and only where it happened to fuse two syllables; it now costs nothing in SER and is counted in CER.
- Tsek bstar (U+0F0C) is a syllable boundary. In 1.2.0 it was not, unless folded with `normalize=True`; folding now affects CER only.
- `bert_segment(max_syllables=...)` windows by slicing the original string, so each window keeps its own shads and spacing, instead of rejoining syllables with a tsek.

### Reproducing 1.2.0 SER

```python
from tibetan_wer import score_segments
split = lambda t: [s for s in t.split("་") if s]
score_segments([split(p) for p in predictions], [split(r) for r in references])
```

## 1.2.0

Rates published with the paper are unchanged: `wer`, `botok_wer`, `ser` and `bert_wer` return the same micro and macro rates and the same edit-operation totals as 1.1.1 on the same input, verified line by line against 1.1.1 on real OCR output.

### Added

- `cer`, the character error rate.
- `per_utterance` (numpy array, aligned with the input) and `per_sentence` (edit counts, reference length, score) in every result.
- `bootstrap_ci`, with `statistic="macro"` or `"micro"`; `cis_overlap`; `significance_matrix`.
- `compare`, a paired bootstrap and permutation test on the difference between two systems scored on the same sentences.
- `error_profile`, `concentration` and `gini`: operation inventory, tsek-boundary edits, space-producing edits, confusion pairs, malformed-syllable share against a vocabulary, and error concentration across lines.
- `edit_operations`, `align`, `edit_counts`, `distance`, `EditOperation`: the alignment itself, not just its totals.
- `repetition_ratio` and `is_degenerate`, the degenerate-output test behind the Gemini filter.
- Public segmenters — `syllable_segment`, `word_segment`, `bert_segment`, `gemini_segment` — plus `segment_all` with per-string caching and, for the API segmenter, `workers` for concurrency.
- `score_segments`, to score text you segmented yourself.
- `normalize` / `normalize_all`, and a `normalize=` argument on every metric: NFC, tsek bstar to tsek, sbrul shad to shad, strip.
- An optional fast alignment backend (`pip install "tibetan-wer[fast]"`, reported by `available_backends`), and `detail="rates"`, which skips operation counts and runs roughly 700x quicker on long lines with identical rates.
- Segmentation fallbacks are counted: `fallback_counts`, `reset_fallback_counts`, and `num_segmentation_fallbacks` in the metric results. `on_error="raise"` opts out of falling back.
- `max_syllables` on `bert_segment` / `bert_wer`, windowing long lines to stay inside the model's 512-token limit.
- A `tibetan-wer` command-line entry point, and an `evaluate` metric module under `integrations/huggingface/`.
- `nlp=` on `bert_wer` and `client=` on `gemini_wer`, to reuse a loaded pipeline or a built API client across calls.
- A test suite (97 tests over `test/`, 92% line coverage), CI across Python 3.9-3.13 run with and without the optional fast backend, and `py.typed`.

### Changed

- Rates are `nan`, not `inf`, when no sentence could be scored.
- `gemini_wer` filters degenerate strings by default (`max_repetition_ratio=10.0`); pass `None` for 1.1.1 behaviour. Filtered sentences are `nan` and counted in `num_skipped`.
- `word_segment` falls back to a syllable split on a string that makes botok raise, instead of propagating; the fallback is counted.
- Minimum Python is 3.9.

### Fixed

- `segment_all` built the segmenter even when the caller supplied one: an API client under `method="gemini", workers>1`, and the BERT pipeline under `method="bert"`, so `nlp=` still required torch to be installed. Both found by tests; never released.

## 1.1.1

Citation and version updated for the published paper (IEEE Access, vol. 14, 2026).

## 1.1.0

BERT and Gemini segmentation (`bert_wer`, `gemini_wer`); `ser`.

## 1.0.0

`wer` / `botok_wer` and `word_segment`.

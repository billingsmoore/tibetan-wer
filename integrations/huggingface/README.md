---
title: Tibetan WER
tags:
- evaluate
- metric
description: >-
  Character (CER), syllable (SER) and segmented word (SWER) error rates for
  Tibetan text.
sdk: static
pinned: false
---

# Metric card: tibetan-wer

Wraps the [`tibetan-wer`](https://github.com/billingsmoore/tibetan-wer) package as an `evaluate` metric.

## Use

```python
import evaluate

metric = evaluate.load("integrations/huggingface/tibetan_wer.py")
metric.compute(predictions=preds, references=refs, metric="ser")
# {'micro': 0.142, 'macro': 0.151, 'substitutions': 812, ...}
```

`metric=` selects the unit: `"cer"` (codepoints), `"ser"` (tsek-delimited syllables), `"wer"`/`"botok"`, `"bert"`, `"gemini"` (the three SWER segmenters). `normalize=True` applies NFC and folds tsek bstar and sbrul shad. `detail="rates"` skips the edit-operation counts and runs far faster on long lines, with identical rates. Extra keyword arguments reach the underlying function — `device=` for `"bert"`, `api_key=` and `workers=` for `"gemini"`.

## What it returns

`micro` weights every unit equally (total edits over total reference units); `macro` weights every sentence equally. Report both — they diverge when line lengths vary. `num_skipped` counts sentences that were not scored (empty reference, or dropped by the Gemini repetition filter); a rate over the remainder is not comparable with one over all sentences.

## Which metric

Tibetan marks syllable boundaries with the tsek (་) and does not mark word boundaries. CER is sub-syllabic and forgiving of errors that destroy a word; word-level evaluation needs automatic segmentation, which SWER supplies. The three segmenters agree on system rankings while disagreeing substantially per line, so pick one, report which, and do not compare SWER figures across segmenters.

## Citation

> J. Moore, S. Li and P. Lauren, "Evaluating Tibetan ASR With Segmented Word Error Rate: Beyond Character-Level Metrics," *IEEE Access*, vol. 14, pp. 101790-101805, 2026.

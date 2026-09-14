# Tibetan-WER

Character (CER), syllable (SER) and segmented word (SWER) error rates for Tibetan text, with three word segmentation methods, per-sentence scores, paired significance tests, error-structure diagnostics and a command-line interface.

This package is the reference implementation of Segmented Word Error Rate (SWER), introduced in:

> J. Moore, S. Li and P. Lauren, "Evaluating Tibetan ASR With Segmented Word Error Rate: Beyond Character-Level Metrics," in *IEEE Access*, vol. 14, pp. 101790-101805, 2026, doi: [10.1109/ACCESS.2026.3709206](https://doi.org/10.1109/ACCESS.2026.3709206).

SWER computes WER for Tibetan by first applying automatic word segmentation to both hypothesis and reference text, since Tibetan orthography marks syllable (*tsek*) boundaries but not word boundaries. Three segmentation methods are provided, each corresponding to a variant reported in the paper.

## Install

```bash
pip install tibetan-wer                  # cer, ser, botok wer, statistics, CLI
pip install "tibetan-wer[fast]"          # + the fast alignment backend
pip install "tibetan-wer[bert]"          # + BERT segmentation
pip install "tibetan-wer[gemini]"        # + Gemini segmentation
```

## Metrics

| Function | Metric | Unit | Extra dependency |
|---|---|---|---|
| `cer` | CER | Unicode codepoint | *(none)* |
| `ser` | SER | tsek (་) delimited syllable | *(none)* |
| `wer` / `botok_wer` | BoTok-SWER | [botok](https://github.com/Esukhia/botok) morphological tokenizer | *(none)* |
| `bert_wer` | BERT-SWER | [KoichiYasuoka/tibetan-bert-base-upos](https://huggingface.co/KoichiYasuoka/tibetan-bert-base-upos) | `tibetan-wer[bert]` |
| `gemini_wer` | Gem-SWER | Gemini 2.5 Flash Lite | `tibetan-wer[gemini]` |
| `score_segments` | any | units you supply | *(none)* |

All take `(predictions, references)`, either as single strings or as equal-length lists, and return a dict:

| Key | |
|---|---|
| `micro_wer` / `macro_wer` | corpus rate, and mean of per-sentence rates (`micro_cer`/`macro_cer` for `cer`, `micro_ser`/`macro_ser` for `ser`) |
| `substitutions`, `insertions`, `deletions` | corpus edit-operation totals |
| `num_sentences`, `num_scored`, `num_skipped` | sentence counts |
| `per_utterance` | numpy array of per-sentence rates, aligned with the input, `nan` where the sentence was not scored |
| `per_sentence` | per-sentence `substitutions`, `insertions`, `deletions`, `edits`, `reference_length`, `score` |
| `num_segmentation_fallbacks` | strings the segmenter could not handle (SWER variants only) |

Note the argument order: `(predictions, references)`. Reversing it leaves substitutions unchanged but swaps insertions and deletions.

A sentence is not scored when its reference is empty, or when the Gemini repetition filter drops it; those appear as `nan` and are excluded from both rates.

## Usage

### CER and SER

```python
from tibetan_wer import cer, ser

predictions = ['གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']
references  = ['འཇམ་དཔལ་གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']

print(f'Micro-CER: {cer(predictions, references)["micro_cer"]:.3f}')
print(f'Micro-SER: {ser(predictions, references)["micro_ser"]:.3f}')
```

### WER

```python
from tibetan_wer import wer, bert_wer, gemini_wer

result = wer(predictions, references)             # botok
result = bert_wer(predictions, references)        # auto-detects CUDA
result = gemini_wer(predictions, references)      # GEMINI_API_KEY, or api_key=

print(f'Micro-WER: {result["micro_wer"]:.3f}')
print(f'Substitutions: {result["substitutions"]}')
```

### Normalization

Nothing is normalized by default: text is scored as stored, no Unicode normalization, no case folding, no punctuation stripping. Rates depend on those conventions, which is a known source of incomparable CER figures, so state which you used.

```python
from tibetan_wer import cer, normalize

cer(predictions, references, normalize=True)      # NFC; ༌ -> ་; ༈ -> །; strip
cer(predictions, references, normalize={'fold_sbrul_shad': False})

normalize('ཀ༌ཁ')      # 'ཀ་ཁ'
```

Folding tsek bstar matters for SER in particular, which splits on the tsek only.

### Confidence intervals

`bootstrap_ci` resamples sentences and recomputes the rate. `statistic='macro'` (default) takes the mean of the per-sentence rates; `statistic='micro'` recomputes total edits over total reference units, matching the reported `micro_*`.

```python
from tibetan_wer import ser, bootstrap_ci, significance_matrix

result = ser(predictions, references)
mean, lo, hi = bootstrap_ci(result)
print(f'SER {mean:.3f} [{lo:.3f}, {hi:.3f}]')

cis = {name: bootstrap_ci(ser(p, references)) for name, p in systems.items()}
significance_matrix(cis)['baseline']['finetuned']   # True if the intervals miss each other
```

### Comparing two systems

Non-overlapping intervals is a conservative test — it ignores that both systems saw the same sentences. `compare` pairs them, which removes the sentence-to-sentence variance that dominates either interval alone:

```python
from tibetan_wer import ser, compare

a, b = ser(preds_a, references), ser(preds_b, references)
c = compare(a, b)

print(f"{c['delta']:+.4f}  95% CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]")
print(c['p_bootstrap'], c['p_permutation'], c['significant'])
```

`delta` is a minus b, so negative means `a` is better. `method='bootstrap'` resamples sentences; `method='permutation'` swaps the two systems' scores within randomly chosen sentences, the exact null of "these systems are interchangeable"; the default runs both. `statistic='micro'` compares micro rates instead.

The two tests disagree in exactly the direction you would expect. Over 144 system pairs from the OCR study, 5 had overlapping intervals but a paired difference that excludes zero — for instance TrOCR-base standard (SER 0.377 [0.369, 0.387]) against its 8-bit quantization (0.395 [0.387, 0.406]) on woodblock print: the intervals overlap, while the paired difference is −0.0178 [−0.0205, −0.0151].

### Per-sentence scores

`per_utterance` is aligned with the input, so it can be correlated against another metric, sorted to find the worst lines, or fed to any further analysis without realigning.

```python
import numpy as np
from tibetan_wer import cer, ser

c = cer(predictions, references)["per_utterance"]
s = ser(predictions, references)["per_utterance"]

ok = ~np.isnan(c) & ~np.isnan(s)
print(np.corrcoef(c[ok], s[ok])[0, 1])          # do the two metrics agree line by line?
print(np.argsort(-np.nan_to_num(c))[:10])       # the ten worst lines
```

### Error structure

A rate says how much a system is wrong; `error_profile` says how, from the same alignment.

```python
from tibetan_wer import error_profile, syllable_segment

p = error_profile(predictions, references, level='char')
p['operations']['deletion_share']      # 0.594 -- this system drops units
p['boundary_edits']['share']           # 0.093 -- edits landing on the tsek
p['space_edits']['share']              # 0.463 -- edits producing a space
p['confusions'][0]                     # {'reference': '་', 'hypothesis': ' ', 'count': 420}
p['concentration']                     # clean-line share, p90, worst-decile share, gini

vocabulary = {s for r in references for s in syllable_segment(r)}
q = error_profile(predictions, references, level='syllable', vocabulary=vocabulary)
q['malformed']['share']                # 0.449 -- share of produced syllables not attested
```

Malformed syllables are mechanically detectable and so correctable without a language model; well-formed but wrong ones are not. Two systems at the same SER can sit at 45% and 10%, which is a difference in how repairable their output is.

### Edit operations

```python
from tibetan_wer import edit_operations

for op in edit_operations(prediction, reference, level='syllable'):
    print(op.op, op.ref, '->', op.hyp, 'at', op.ref_index)
    # S ནུར -> ནུས at 3
```

`level` is `'char'`, `'syllable'`, `'botok'`, `'bert'` or `'gemini'`. Lower-level entry points — `align`, `edit_counts`, `distance` — take unit sequences directly.

### Segmenting once, scoring many

Segmentation, not alignment, dominates the cost of an evaluation with the BERT or Gemini segmenter. Segment once, then score every system against the same segmentation:

```python
from tibetan_wer import segment_all, score_segments

cache = {}                                       # {text: tokens}, reusable, picklable
ref_segments = segment_all(references, method='bert', cache=cache)

for name, preds in systems.items():
    hyp_segments = segment_all(preds, method='bert', cache=cache)
    print(name, score_segments(hyp_segments, ref_segments)['micro_wer'])
```

`method` is `'syllable'`, `'botok'`, `'bert'` or `'gemini'`; each distinct string is segmented once. For the API segmenter, `workers=8` runs calls concurrently. The individual segmenters — `syllable_segment`, `word_segment`, `bert_segment`, `gemini_segment` — are also public.

### Speed

The alignment ships in two backends. `python` is the reference implementation whose backtrace produced the published operation counts. `fast` uses `rapidfuzz` and is roughly 700x quicker, returning **the same distance on every input** — so rates are identical — but where several alignments tie in cost it may split a total differently between substitutions, insertions and deletions.

So rates and operation counts are treated differently:

```python
cer(predictions, references)                     # exact counts, reference alignment
cer(predictions, references, detail='rates')     # identical rates, ~700x faster, no S/I/D
cer(predictions, references, backend='fast')     # fast counts, may redistribute S/I/D
```

`detail='rates'` is the one to reach for when scoring a grid of systems, or bootstrapping: on 400 lines of woodblock OCR, CER takes 7.6 s at `detail='full'` and 0.010 s at `detail='rates'`, with identical micro and macro rates.

### When a segmenter fails

botok can raise on pathological input, and the BERT model reads at most 512 tokens. A failure falls back to a syllable split for that string, which silently turns that line's word rate into a syllable rate — so it is counted rather than swallowed:

```python
result = wer(predictions, references)
result['num_segmentation_fallbacks']             # 0, or a caveat on the number

from tibetan_wer import fallback_counts
fallback_counts()                                # {'botok': 3} -- by segmenter

wer(predictions, references, on_error='raise')   # or refuse to fall back at all
bert_wer(predictions, references, max_syllables=100)   # window long lines instead
```

### Degenerate output

A collapsed model emits long runs of a few characters. Such strings make the API segmenter time out rather than merely score badly, so `gemini_wer` drops any sentence whose prediction or reference has a character repetition ratio (length ÷ distinct characters) above `max_repetition_ratio`, default 10. Dropped sentences are `nan` in `per_utterance` and counted in `num_skipped`.

```python
result = gemini_wer(predictions, references, max_repetition_ratio=10.0, workers=8)
print(result['num_skipped'], 'of', result['num_sentences'], 'sentences not scored')
```

A rate computed over the remainder is not comparable with one computed over every sentence, and the dropped sentences are ones the system did badly on, so report `num_skipped` alongside the rate. Pass `max_repetition_ratio=None` to disable the filter, and `max_output_tokens=` to cap each response.

## Command line

```bash
tibetan-wer predictions.txt references.txt --metric ser --ci
tibetan-wer a.txt references.txt --compare b.txt --metric wer
tibetan-wer predictions.txt references.txt --profile --level syllable --vocabulary -
tibetan-wer predictions.txt references.txt --metric cer --detail rates --json out.json
```

Files hold one sentence per line, UTF-8, prediction *i* against reference *i*; `-` reads predictions from stdin. `--vocabulary -` builds the attested-syllable set from the references themselves.

```
SER over 300 sentences
  micro  0.1011
  macro  0.1099
  95% CI [0.0974, 0.1232]  (macro rate, bootstrap)
  S/I/D  793 / 44 / 34

comparison (macro rate, paired over 300 sentences)
  delta -0.0719  95% CI [-0.0923, -0.0534]
  p (bootstrap) 0.0000   p (permutation) 0.0001
  significant at the 5% level
```

## Hugging Face `evaluate`

`integrations/huggingface/` holds an `evaluate` metric module:

```python
import evaluate
metric = evaluate.load("integrations/huggingface/tibetan_wer.py")
metric.compute(predictions=preds, references=refs, metric="ser")
```

## Usage for Model Evaluation

```python
from tibetan_wer import cer as tib_cer, wer as tib_wer, ser as tib_ser

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids,  skip_special_tokens=True)

    # detail='rates' keeps evaluation off the critical path during training
    cer_result = tib_cer(pred_str, label_str, detail="rates")
    wer_result = tib_wer(pred_str, label_str, detail="rates")
    ser_result = tib_ser(pred_str, label_str, detail="rates")

    return {
        "tib_micro_cer": cer_result["micro_cer"],
        "tib_macro_cer": cer_result["macro_cer"],
        "tib_micro_wer": wer_result["micro_wer"],
        "tib_macro_wer": wer_result["macro_wer"],
        "tib_micro_ser": ser_result["micro_ser"],
        "tib_macro_ser": ser_result["macro_ser"],
    }
```

```python
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

## Reference

Everything importable from `tibetan_wer`:

| | |
|---|---|
| **Metrics** | |
| `cer(predictions, references, normalize=False, backend='auto', detail='full')` | character error rate |
| `ser(...)` | syllable error rate, tsek-delimited |
| `wer(..., cache=None, on_error='fallback')` / `botok_wer` | BoTok-SWER |
| `bert_wer(..., device=None, max_syllables=None, nlp=None)` | BERT-SWER; `nlp=` reuses a loaded pipeline |
| `gemini_wer(..., api_key=None, max_repetition_ratio=10.0, max_output_tokens=None, workers=1, client=None)` | Gem-SWER; `client=` reuses a built client |
| `score_segments(prediction_segments, reference_segments, unit='wer')` | score units you segmented yourself |
| **Statistics** | |
| `bootstrap_ci(scores, n_iterations=10000, percentiles=(2.5, 97.5), seed=42, statistic='macro')` | `(point, lo, hi)` |
| `compare(result_a, result_b, statistic='macro', method='both')` | paired bootstrap and permutation test |
| `cis_overlap(ci_a, ci_b)` | do two `(lo, hi)` intervals meet |
| `significance_matrix({name: (mean, lo, hi)})` | pairwise non-overlap |
| **Diagnostics** | |
| `error_profile(predictions, references, level='char', vocabulary=None, top_confusions=10)` | operations, boundary and space edits, confusions, malformed share, concentration |
| `concentration(per_utterance)` | clean-line share, p90, worst-decile share, gini |
| `gini(values)` | Gini coefficient of a non-negative array |
| `edit_operations(prediction, reference, level='char')` | the edits between one pair |
| **Alignment** | |
| `align(ref_units, hyp_units, backend='auto')` | list of `EditOperation` |
| `edit_counts(ref_units, hyp_units)` | `(S, I, D)` |
| `distance(ref_units, hyp_units)` | Levenshtein distance |
| `EditOperation` | `(op, ref, hyp, ref_index, hyp_index)`, `op` in `S`/`I`/`D` |
| `available_backends()` | `['fast', 'python']` or `['python']` |
| **Segmentation** | |
| `syllable_segment(text)` | split on the tsek |
| `word_segment(sentence, tokenizer=None, on_error='fallback')` / `botok_segment` | botok |
| `bert_segment(text, device=None, max_syllables=None, on_error='fallback')` | UPOS classifier |
| `gemini_segment(text, api_key=None, model=..., max_output_tokens=None, on_error='raise')` | Gemini |
| `segment_all(texts, method='botok', cache=None, workers=1, **kwargs)` | batch, de-duplicated |
| `fallback_counts()` / `reset_fallback_counts()` | strings that fell back to a syllable split |
| `repetition_ratio(text)` / `is_degenerate(text, threshold=10.0)` | degenerate-output detection |
| **Normalization** | |
| `normalize(text, form='NFC', fold_tsek_bstar=True, fold_sbrul_shad=True, strip=True)` | one string |
| `normalize_all(texts, **kwargs)` | many |

### Command-line options

| Option | |
|---|---|
| `--metric {cer,ser,wer,bert,gemini}` | default `ser` |
| `--compare FILE` | paired comparison against a second system |
| `--ci`, `--statistic {macro,micro}`, `--iterations N`, `--seed N` | bootstrap interval |
| `--normalize` | NFC, fold tsek bstar and sbrul shad, strip |
| `--detail {full,rates}`, `--backend {auto,python,fast}` | alignment cost and exactness |
| `--profile`, `--level {char,syllable,botok,bert,gemini}`, `--vocabulary FILE` | error structure; `--vocabulary -` builds one from the references |
| `--workers N` | concurrent API calls for `--metric gemini` |
| `--json [FILE]` | JSON instead of text, stdout when no file is given |

## Tests

```bash
pip install "tibetan-wer[dev]"
pytest test/                       # or run any file directly: python test/test_cli.py
```

Three files, 97 tests, 92% line coverage:

| File | |
|---|---|
| `test/test_metrics.py` | metrics, alignment, backends, normalization, statistics, diagnostics, and a cross-check of `score_segments` against `jiwer` on space-delimited text |
| `test/test_segmenter_plumbing.py` | the BERT and Gemini paths — caching, windowing, the repetition filter, fallback accounting, concurrency — with stand-ins passed through `nlp=` and `client=` |
| `test/test_cli.py` | the command line: text and JSON reports, every `--metric` branch, the vocabulary options, stdin, exit statuses |

The two segmenters that need a model download or a paid endpoint are never called for real, and CI runs the suite twice, with and without the optional fast backend.

## Changes

See [CHANGELOG.md](CHANGELOG.md). Rates published with the paper are unchanged in 1.2.0.

## Citation

If you use this package, please cite:

```bibtex
@ARTICLE{moore2026tibetanasr,
  author={Moore, Jacob and Li, Sheng and Lauren, Paula},
  journal={IEEE Access},
  title={Evaluating Tibetan {ASR} With Segmented Word Error Rate: Beyond Character-Level Metrics},
  year={2026},
  volume={14},
  pages={101790-101805},
  doi={10.1109/ACCESS.2026.3709206}
}
```

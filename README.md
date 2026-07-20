# Tibetan-WER

Word Error Rate (WER) and Syllable Error Rate (SER) metrics for Tibetan ASR evaluation, with three word segmentation methods.

This package is the reference implementation of Segmented Word Error Rate (SWER), introduced in:

> J. Moore, S. Li and P. Lauren, "Evaluating Tibetan ASR With Segmented Word Error Rate: Beyond Character-Level Metrics," in *IEEE Access*, vol. 14, pp. 101790-101805, 2026, doi: [10.1109/ACCESS.2026.3709206](https://doi.org/10.1109/ACCESS.2026.3709206).

SWER computes WER for Tibetan by first applying automatic word segmentation to both hypothesis and reference text, since Tibetan orthography marks syllable (*tsek*) boundaries but not word boundaries. Three segmentation methods are provided, each corresponding to a variant reported in the paper.

## Install

```bash
pip install tibetan-wer
```

For BERT-based segmentation:

```bash
pip install "tibetan-wer[bert]"
```

For Gemini-based segmentation:

```bash
pip install "tibetan-wer[gemini]"
```

## Functions

| Function | SWER variant | Segmentation method | Extra dependency |
|---|---|---|---|
| `wer` / `botok_wer` | BoTok-SWER | [botok](https://github.com/Esukhia/botok) morphological tokenizer | *(none)* |
| `ser` | — | tsek (་) syllable boundary | *(none)* |
| `bert_wer` | BERT-SWER | [KoichiYasuoka/tibetan-bert-base-upos](https://huggingface.co/KoichiYasuoka/tibetan-bert-base-upos) | `tibetan-wer[bert]` |
| `gemini_wer` | Gem-SWER | Gemini 2.5 Flash Lite | `tibetan-wer[gemini]` |

All functions accept either a single string or a list of strings and return a dict with `micro_wer`/`macro_wer` (or `micro_ser`/`macro_ser`), plus `substitutions`, `insertions`, `deletions`, and `num_sentences`.

## Usage

### WER (botok)

```python
from tibetan_wer import wer

predictions = ['གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']
references  = ['འཇམ་དཔལ་གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']

result = wer(predictions, references)

print(f'Micro-WER: {result["micro_wer"]:.3f}')
print(f'Macro-WER: {result["macro_wer"]:.3f}')
print(f'Substitutions: {result["substitutions"]}')
print(f'Insertions:    {result["insertions"]}')
print(f'Deletions:     {result["deletions"]}')
```

### SER

```python
from tibetan_wer import ser

result = ser(predictions, references)

print(f'Micro-SER: {result["micro_ser"]:.3f}')
print(f'Macro-SER: {result["macro_ser"]:.3f}')
```

### BERT WER

```python
from tibetan_wer import bert_wer

result = bert_wer(predictions, references)          # auto-detects CUDA
result = bert_wer(predictions, references, device=0)  # force GPU 0
```

### Gemini WER

```python
from tibetan_wer import gemini_wer

result = gemini_wer(predictions, references)
# api_key defaults to the GEMINI_API_KEY environment variable
result = gemini_wer(predictions, references, api_key="YOUR_KEY")
```

## Usage for Model Evaluation

```python
import evaluate
from tibetan_wer import wer as tib_wer, ser as tib_ser

cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str  = tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids,  skip_special_tokens=True)

    cer         = cer_metric.compute(predictions=pred_str, references=label_str)
    wer_result  = tib_wer(pred_str, label_str)
    ser_result  = tib_ser(pred_str, label_str)

    return {
        "cer":                    cer,
        "tib_macro_wer":          wer_result["macro_wer"],
        "tib_micro_wer":          wer_result["micro_wer"],
        "word_substitutions":     wer_result["substitutions"],
        "word_insertions":        wer_result["insertions"],
        "word_deletions":         wer_result["deletions"],
        "tib_macro_ser":          ser_result["macro_ser"],
        "tib_micro_ser":          ser_result["micro_ser"],
        "syllable_substitutions": ser_result["substitutions"],
        "syllable_insertions":    ser_result["insertions"],
        "syllable_deletions":     ser_result["deletions"],
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

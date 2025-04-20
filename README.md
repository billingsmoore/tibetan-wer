# Tibetan-WER

This module provides a means to calculate Word Error Rate, and the Syllable Error Rate for Tibetan language text.

## Install

Install the library to get started:

```bash
pip install --upgrade tibetan_wer
```

## Usage

### Basic Usage

The `wer` function expects a list of predictions and a list of references and returns a dictionary of the micro and macro average WER as well as the total number of substitutions, insertions, and deletions.

```python
from tibetan_wer.metrics import wer

rediction = ['གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']
reference = ['འཇམ་དཔལ་གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']

result = wer(prediction, reference)

print(f'Micro-Average WER Score: {result['micro_wer']}')
print(f'Macro-Average WER Score: {result['macro_wer']}')
print(f'Substitutions: {result['substitutions']}')
print(f'Insertions: {result['insertions']}')
print(f'Deletions: {result['deletions']}')
```

The `ser` function works very similarly.

```python
from tibetan_wer.metrics import ser

prediction = ['གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']
reference = ['འཇམ་དཔལ་གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']

result = ser(prediction, reference)

print(f'Micro-Average SER Score: {result['micro_ser']:.3f}')
print(f'Macro-Average SER Score: {result['macro_ser']:.3f}')
print(f'Substitutions: {result['substitutions']:.3f}')
print(f'Insertions: {result['insertions']:.3f}')
print(f'Deletions: {result['deletions']:.3f}')
```

### Usage for Model Evaluation

The intended use-case is as part of assessing model training. To use `tibetan_wer` for this you can define custom metrics for model training like so:

```python
import evaluate
from tibetan_wer.metrics import wer as tib_wer, ser as tib_ser

cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    tib_wer_res = tib_wer(predictions=pred_str, references=label_str)
    tib_ser_res = tib_ser(predictions=pred_str, references=label_str)

    macro_wer = tib_wer_res['macro_wer']
    micro_wer = tib_wer_res['micro_wer']
    word_subs = tib_wer_res['substitutions']
    word_ins = tib_wer_res['insertions']
    word_dels = tib_wer_res['deletions']

    macro_ser = tib_ser_res['macro_ser']
    micro_ser = tib_ser_res['micro_ser']
    syl_subs = tib_ser_res['substitutions']
    syl_ins = tib_ser_res['insertions']
    syl_dels = tib_ser_res['deletions']

    return {"cer": cer,
            "tib_macro_wer": macro_wer,
            "tib_micro_wer": micro_wer,
            "word_substitutions": word_subs,
            "word_insertions":word_ins,
            "word_deletions":word_dels,
            "tib_macro_ser": macro_ser,
            "tib_micro_ser": micro_ser,
            "syllable_substitutions": syl_subs,
            "syllable_insertions": syl_ins,
            "syllable_deletions": syl_dels
            }
```

You can then set the `transformers` trainer to use these metrics like so:

```python
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics, # use custom metrics
    tokenizer=processor.feature_extractor,
)

trainer.train()
```

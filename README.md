# Tibetan-WER

This module provides a means to calculate Word Error Rate for Tibetan language text.

## Install

Install the library to get started:

```bash
pip install --upgrade tibetan_wer
```

## Usage

The `wer` function expects a list of predictions and a list of references and returns a dictionary of the micro and macro average WER as well as the total number of substitutions, insertions, and deletions.

```python
from tibetan_wer.wer import wer

rediction = ['གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']
reference = ['འཇམ་དཔལ་གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ༔']

result = wer(prediction, reference)

print(f'Micro-Average WER Score: {result['micro_wer']}')
print(f'Macro-Average WER Score: {result['macro_wer']}')
print(f'Substitutions: {result['substitutions']}')
print(f'Insertions: {result['insertions']}')
print(f'Deletions: {result['deletions']}')
```
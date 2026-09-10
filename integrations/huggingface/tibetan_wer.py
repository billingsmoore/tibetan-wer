"""Tibetan error rates as an `evaluate` metric module.

Load it straight from this directory or from the Hub:

    import evaluate
    metric = evaluate.load("integrations/huggingface/tibetan_wer.py")
    metric.compute(predictions=preds, references=refs, metric="ser")
"""
import datasets
import evaluate

import tibetan_wer

_CITATION = """\
@ARTICLE{moore2026tibetanasr,
  author={Moore, Jacob and Li, Sheng and Lauren, Paula},
  journal={IEEE Access},
  title={Evaluating Tibetan {ASR} With Segmented Word Error Rate: Beyond Character-Level Metrics},
  year={2026},
  volume={14},
  pages={101790-101805},
  doi={10.1109/ACCESS.2026.3709206}
}
"""

_DESCRIPTION = """\
Character (CER), syllable (SER) and segmented word (SWER) error rates for Tibetan.

Tibetan orthography marks syllable boundaries with the tsek but does not mark
word boundaries, so word-level evaluation requires automatic segmentation.
SWER applies the same segmenter to hypothesis and reference before computing
WER. Three segmenters are available: botok (rule-based), a BERT UPOS token
classifier, and Gemini.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: list of str, the transcriptions to score.
    references: list of str, the ground truth.
    metric: one of "cer", "ser", "wer" (botok), "bert", "gemini". Default "ser".
    normalize: bool or dict. False (default) scores text as stored; True applies
        NFC, folds tsek bstar and sbrul shad, and strips whitespace.
    detail: "full" (default) reports substitutions/insertions/deletions;
        "rates" skips them and is far faster on long lines, with identical rates.
    **kwargs: passed through to the underlying function, e.g. device= for
        "bert", api_key= and workers= for "gemini".

Returns:
    micro: float, corpus rate (total edits over total reference units).
    macro: float, mean of the per-sentence rates.
    substitutions, insertions, deletions: int, corpus totals ("full" only).
    num_sentences, num_scored, num_skipped: int.

Examples:
    >>> metric = evaluate.load("tibetan_wer")
    >>> metric.compute(predictions=["ཀ་ཁ་"], references=["ཀ་ག་"], metric="ser")
"""

_FUNCTIONS = {
    "cer": "cer",
    "ser": "ser",
    "wer": "wer",
    "botok": "wer",
    "bert": "bert_wer",
    "gemini": "gemini_wer",
}

_UNITS = {"cer": "cer", "ser": "ser", "wer": "wer", "botok": "wer", "bert": "wer", "gemini": "wer"}


class TibetanWER(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/billingsmoore/tibetan-wer"],
            reference_urls=["https://doi.org/10.1109/ACCESS.2026.3709206"],
        )

    def _compute(self, predictions, references, metric="ser", normalize=False, detail="full", **kwargs):
        if metric not in _FUNCTIONS:
            raise ValueError(
                "unknown metric {!r}; expected one of {}".format(metric, sorted(_FUNCTIONS))
            )
        function = getattr(tibetan_wer, _FUNCTIONS[metric])
        result = function(predictions, references, normalize=normalize, detail=detail, **kwargs)
        unit = _UNITS[metric]
        out = {
            "micro": result["micro_" + unit],
            "macro": result["macro_" + unit],
            "num_sentences": result["num_sentences"],
            "num_scored": result["num_scored"],
            "num_skipped": result["num_skipped"],
        }
        for key in ("substitutions", "insertions", "deletions", "num_segmentation_fallbacks"):
            if result.get(key) is not None:
                out[key] = result[key]
        return out

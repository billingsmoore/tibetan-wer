"""Tests for the BERT and Gemini paths, using stand-ins for the model and the API.

Neither segmenter can be exercised in CI -- one downloads a model, the other
needs a paid endpoint -- but everything around them can: caching, windowing,
fallback accounting, the repetition filter, concurrency, and the argument
plumbing that carries options down to the segmenter. The stand-ins here are
passed through the same public parameters a caller would use (``nlp=``,
``client=``), so the code under test is the real path.

Runs under pytest, or standalone with `python test/test_segmenter_plumbing.py`.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tibetan_wer.segmentation as seg  # noqa: E402
from tibetan_wer import (  # noqa: E402
    bert_wer,
    fallback_counts,
    gemini_wer,
    reset_fallback_counts,
    segment_all,
)

REF = "འཇམ་དཔལ་གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ"
HYP = "འཇམ་དཔལ་གཞོན་ནུས་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ"
GARBAGE = "ཀཀཀཀཀཀཀཀཀཀཀཀཀཀཀཀཀཀཀཀ"


# --- stand-ins -------------------------------------------------------------

class FakePipeline:
    """Stands in for the transformers token-classification pipeline."""

    def __init__(self, fail_on=(), empty_on=()):
        self.calls = []
        self.fail_on = set(fail_on)
        self.empty_on = set(empty_on)

    def __call__(self, text):
        self.calls.append(text)
        if text in self.fail_on:
            raise RuntimeError("pipeline exploded")
        if text in self.empty_on:
            return []
        # pair up syllables, so the segmentation differs from a syllable split
        syllables = seg.syllable_segment(text)
        return [
            {"word": "".join(syllables[i:i + 2])}
            for i in range(0, len(syllables), 2)
        ]


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    def __init__(self, client):
        self._client = client

    def generate_content(self, model=None, contents=None, config=None):
        self._client.calls.append({"model": model, "contents": contents, "config": config})
        text = contents.rsplit("Text: ", 1)[-1]
        if text in self._client.fail_on:
            raise RuntimeError("API said no")
        return FakeResponse("|".join(seg.syllable_segment(text)))


class FakeGemini:
    """Stands in for google.genai.Client."""

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)
        self.models = FakeModels(self)


# --- BERT ------------------------------------------------------------------

def test_bert_segment_pairs_syllables_through_the_pipeline():
    nlp = FakePipeline()
    assert seg.bert_segment("ཀ་ཁ་ག་ང", nlp=nlp) == ["ཀཁ", "གང"]
    assert nlp.calls == ["ཀ་ཁ་ག་ང"]


def test_bert_segment_returns_nothing_for_blank_input():
    nlp = FakePipeline()
    assert seg.bert_segment("   ", nlp=nlp) == []
    assert nlp.calls == []


def test_bert_segment_falls_back_and_counts_on_failure():
    reset_fallback_counts()
    nlp = FakePipeline(fail_on={"ཀ་ཁ"})
    assert seg.bert_segment("ཀ་ཁ", nlp=nlp) == ["ཀ", "ཁ"]
    assert fallback_counts()["bert"] == 1


def test_bert_segment_falls_back_when_the_model_returns_nothing():
    reset_fallback_counts()
    nlp = FakePipeline(empty_on={"ཀ་ཁ"})
    assert seg.bert_segment("ཀ་ཁ", nlp=nlp) == ["ཀ", "ཁ"]
    assert fallback_counts()["bert"] == 1


def test_bert_segment_raises_when_told_to():
    nlp = FakePipeline(fail_on={"ཀ་ཁ"})
    try:
        seg.bert_segment("ཀ་ཁ", nlp=nlp, on_error="raise")
    except RuntimeError:
        return
    raise AssertionError("expected the pipeline error to propagate")


def test_max_syllables_windows_long_lines():
    nlp = FakePipeline()
    text = "་".join("ཀཁགངཅཆཇཉ") + "་"          # 8 syllables
    windowed = seg.bert_segment(text, nlp=nlp, max_syllables=4)
    assert len(nlp.calls) == 2                   # two windows, not one long call
    assert all(len(seg.syllable_segment(c)) <= 4 for c in nlp.calls)
    assert "".join(windowed) == "".join(seg.syllable_segment(text))


def test_max_syllables_leaves_short_lines_alone():
    nlp = FakePipeline()
    seg.bert_segment("ཀ་ཁ་", nlp=nlp, max_syllables=100)
    assert len(nlp.calls) == 1


def test_bert_wer_scores_through_an_injected_pipeline():
    nlp = FakePipeline()
    result = bert_wer([HYP], [REF], nlp=nlp)
    assert 0 < result["micro_wer"] <= 1
    assert result["num_segmentation_fallbacks"] == 0
    assert len(nlp.calls) == 2                   # one per distinct string


def test_bert_wer_reports_fallbacks_it_incurred():
    reset_fallback_counts()
    result = bert_wer([HYP], [REF], nlp=FakePipeline(fail_on={HYP}))
    assert result["num_segmentation_fallbacks"] == 1


# --- Gemini ----------------------------------------------------------------

def test_gemini_segment_parses_the_pipe_format():
    client = FakeGemini()
    assert seg.gemini_segment("ཀ་ཁ་ག", client=client, pause=0) == ["ཀ", "ཁ", "ག"]
    assert client.calls[0]["config"]["temperature"] == 0.0
    assert "max_output_tokens" not in client.calls[0]["config"]


def test_gemini_segment_passes_the_token_cap_through():
    client = FakeGemini()
    seg.gemini_segment("ཀ་ཁ", client=client, pause=0, max_output_tokens=64)
    assert client.calls[0]["config"]["max_output_tokens"] == 64


def test_gemini_segment_retries_then_falls_back():
    reset_fallback_counts()
    client = FakeGemini(fail_on={"ཀ་ཁ"})
    assert seg.gemini_segment(
        "ཀ་ཁ", client=client, pause=0, max_retries=1, on_error="fallback"
    ) == ["ཀ", "ཁ"]
    assert fallback_counts()["gemini"] == 1


def test_gemini_segment_raises_by_default():
    client = FakeGemini(fail_on={"ཀ་ཁ"})
    try:
        seg.gemini_segment("ཀ་ཁ", client=client, pause=0, max_retries=1)
    except RuntimeError:
        return
    raise AssertionError("expected the API error to propagate")


def test_gemini_wer_skips_degenerate_strings_without_calling_the_api():
    client = FakeGemini()
    result = gemini_wer([GARBAGE, HYP], [REF, REF], client=client, cache={})
    assert result["num_skipped"] == 1
    assert np.isnan(result["per_utterance"][0])
    assert not np.isnan(result["per_utterance"][1])
    assert all(GARBAGE not in c["contents"] for c in client.calls)


def test_gemini_wer_filter_can_be_disabled():
    client = FakeGemini()
    result = gemini_wer(
        [GARBAGE], [REF], client=client, cache={}, max_repetition_ratio=None
    )
    assert result["num_skipped"] == 0
    assert any(GARBAGE in c["contents"] for c in client.calls)


def test_gemini_wer_reuses_a_supplied_cache():
    client = FakeGemini()
    cache = {}
    gemini_wer([HYP], [REF], client=client, cache=cache)
    first = len(client.calls)
    gemini_wer([HYP], [REF], client=client, cache=cache)
    assert len(client.calls) == first            # nothing re-segmented
    assert set(cache) == {HYP, REF}


# --- batching --------------------------------------------------------------

def test_segment_all_runs_workers_concurrently_and_keeps_order():
    client = FakeGemini()
    texts = ["ཀ་ཁ", "ག་ང", "ཅ་ཆ", "ཇ་ཉ"]
    out = segment_all(texts, method="gemini", workers=4, client=client, pause=0)
    assert out == [seg.syllable_segment(t) for t in texts]
    assert len(client.calls) == 4


def test_segment_all_rejects_an_unknown_method():
    try:
        segment_all(["ཀ་ཁ"], method="telepathy")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS " + name)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("FAIL {}: {}: {}".format(name, type(exc).__name__, exc))
    print("\n{} failure(s)".format(failures))
    sys.exit(1 if failures else 0)

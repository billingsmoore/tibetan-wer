"""Tests for tibetan_wer.

Runs under pytest, or standalone with `python test/test_metrics.py`, which is
useful in environments where only the runtime dependencies are installed.
Tests requiring botok are skipped if it is unavailable; nothing here touches
the network or the BERT/Gemini segmenters.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tibetan_wer import (  # noqa: E402
    align,
    compare,
    bootstrap_ci,
    cer,
    cis_overlap,
    distance,
    edit_counts,
    edit_operations,
    is_degenerate,
    repetition_ratio,
    score_segments,
    segment_all,
    ser,
    significance_matrix,
    syllable_segment,
)

REF = "འཇམ་དཔལ་གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ"
HYP = "གཞོན་ནུར་གྱུར་པ་ལ་ཕྱག་འཚལ་ལོ"


# --- alignment -------------------------------------------------------------

def test_align_reports_each_operation():
    ops = align(list("abc"), list("axc"))
    assert [o.op for o in ops] == ["S"]
    assert ops[0].ref == "b" and ops[0].hyp == "x"
    assert ops[0].ref_index == 1 and ops[0].hyp_index == 1


def test_align_is_left_to_right():
    ops = align(list("abcd"), list("xbcy"))
    assert [(o.op, o.ref_index) for o in ops] == [("S", 0), ("S", 3)]


def test_insertion_and_deletion_carry_one_side_only():
    ins = align(list("ac"), list("abc"))
    assert len(ins) == 1 and ins[0].op == "I" and ins[0].ref is None and ins[0].hyp == "b"
    dele = align(list("abc"), list("ac"))
    assert len(dele) == 1 and dele[0].op == "D" and dele[0].hyp is None and dele[0].ref == "b"


def test_edit_counts_agree_with_distance():
    S, I, D = edit_counts(list(REF), list(HYP))
    assert S + I + D == distance(list(REF), list(HYP))


def test_identical_sequences_have_no_operations():
    assert align(list(REF), list(REF)) == []
    assert edit_counts(list(REF), list(REF)) == (0, 0, 0)


def test_edit_operations_levels():
    char_ops = edit_operations(HYP, REF, level="char")
    syl_ops = edit_operations(HYP, REF, level="syllable")
    assert all(o.op == "D" for o in char_ops)      # the hypothesis drops a prefix
    assert all(o.op == "D" for o in syl_ops)
    assert len(syl_ops) == 2                        # two missing syllables
    assert [o.ref for o in syl_ops] == ["འཇམ", "དཔལ"]


# --- CER -------------------------------------------------------------------

def test_cer_perfect_prediction_is_zero():
    r = cer([REF], [REF])
    assert r["micro_cer"] == 0.0 and r["macro_cer"] == 0.0
    assert (r["substitutions"], r["insertions"], r["deletions"]) == (0, 0, 0)


def test_cer_counts_and_rate():
    r = cer(["abd"], ["abc"])
    assert r["substitutions"] == 1
    assert r["micro_cer"] == 1 / 3
    assert r["num_sentences"] == 1


def test_cer_accepts_bare_strings():
    assert cer(HYP, REF)["micro_cer"] == cer([HYP], [REF])["micro_cer"]


def test_cer_micro_and_macro_differ_with_uneven_lengths():
    r = cer(["ab", "xxxx"], ["ab", "yyyy"])
    assert r["micro_cer"] == 4 / 6
    assert r["macro_cer"] == 0.5


# --- SER -------------------------------------------------------------------

def test_ser_keys_and_value():
    r = ser([HYP], [REF])
    assert set(["micro_ser", "macro_ser"]).issubset(r)
    assert r["deletions"] == 2
    assert r["micro_ser"] == 2 / len(syllable_segment(REF))


def test_ser_argument_order_swaps_insertions_and_deletions():
    forward = ser([HYP], [REF])
    reversed_ = ser([REF], [HYP])
    assert forward["deletions"] == reversed_["insertions"] == 2
    assert forward["insertions"] == reversed_["deletions"] == 0
    assert forward["substitutions"] == reversed_["substitutions"]


def test_syllable_segment_ignores_tsek_bstar():
    assert syllable_segment("ཀ་ཁ") == ["ཀ", "ཁ"]
    assert syllable_segment("ཀ༌ཁ") == ["ཀ༌ཁ"]


# --- per-sentence output ---------------------------------------------------

def test_per_utterance_is_aligned_with_input():
    r = cer(["abc", "xbc"], ["abc", "abc"])
    assert np.allclose(r["per_utterance"], [0.0, 1 / 3])
    assert r["num_scored"] == 2 and r["num_skipped"] == 0


def test_per_sentence_counts_sum_to_totals():
    r = ser([HYP, REF], [REF, HYP])
    for key in ("substitutions", "insertions", "deletions"):
        assert sum(s[key] for s in r["per_sentence"]) == r[key]


def test_empty_reference_is_nan_not_an_error():
    r = cer(["abc", ""], ["", "abc"])
    assert np.isnan(r["per_utterance"][0])
    assert r["num_scored"] == 1
    assert r["macro_cer"] == 1.0


def test_all_empty_references_give_nan_rates():
    r = cer(["abc"], [""])
    assert np.isnan(r["micro_cer"]) and np.isnan(r["macro_cer"])


def test_length_mismatch_raises():
    try:
        cer(["a"], ["a", "b"])
    except ValueError:
        return
    raise AssertionError("expected ValueError on mismatched lengths")


# --- pre-segmented scoring -------------------------------------------------

def test_score_segments_matches_ser_on_the_same_units():
    refs = [syllable_segment(REF)]
    hyps = [syllable_segment(HYP)]
    direct = score_segments(hyps, refs, unit="ser")
    assert direct["micro_ser"] == ser([HYP], [REF])["micro_ser"]


def test_score_segments_skips_none_entries():
    r = score_segments([["a"], None], [["a"], ["b"]])
    assert r["num_skipped"] == 1 and r["num_scored"] == 1
    assert np.isnan(r["per_utterance"][1])
    assert r["micro_wer"] == 0.0


# --- degenerate output -----------------------------------------------------

def test_repetition_ratio_and_threshold():
    assert repetition_ratio("") == 0.0
    assert repetition_ratio("abcd") == 1.0
    assert repetition_ratio("ab" * 20) == 20.0
    assert is_degenerate("ab" * 20)
    assert not is_degenerate(REF)


# --- botok -----------------------------------------------------------------

def _botok_available():
    try:
        import botok  # noqa: F401
        return True
    except Exception:
        return False


def test_segment_all_caches_and_preserves_order():
    calls = []

    def fake(text):
        calls.append(text)
        return text.split()

    import tibetan_wer.segmentation as seg
    seg._METHODS["_fake"] = fake
    try:
        cache = {}
        out = segment_all(["a b", "a b", "c"], method="_fake", cache=cache)
        assert out == [["a", "b"], ["a", "b"], ["c"]]
        assert calls == ["a b", "c"]          # each distinct string segmented once
        segment_all(["a b"], method="_fake", cache=cache)
        assert calls == ["a b", "c"]          # cache reused across calls
    finally:
        del seg._METHODS["_fake"]


def test_botok_wer_runs():
    if not _botok_available():
        print("  (skipped: botok not installed)")
        return
    from tibetan_wer import wer
    r = wer([HYP], [REF])
    assert 0.0 < r["micro_wer"] <= 1.0
    assert r["num_sentences"] == 1
    assert len(r["per_utterance"]) == 1


# --- statistics ------------------------------------------------------------

def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mean, lo, hi = bootstrap_ci(values, n_iterations=2000)
    assert mean == 0.3
    assert lo < mean < hi
    assert (mean, lo, hi) == bootstrap_ci(values, n_iterations=2000)


def test_bootstrap_ci_ignores_nan_and_handles_empty():
    values = np.array([0.2, np.nan, 0.4])
    mean, lo, hi = bootstrap_ci(values, n_iterations=500)
    assert abs(mean - 0.3) < 1e-12
    assert all(np.isnan(v) for v in bootstrap_ci(np.array([np.nan]), n_iterations=10))


def test_bootstrap_ci_consumes_a_metric_result_directly():
    r = cer(["abc", "xbc", "xyc"], ["abc", "abc", "abc"])
    mean, lo, hi = bootstrap_ci(r["per_utterance"], n_iterations=1000)
    assert abs(mean - r["macro_cer"]) < 1e-12


def test_significance_is_non_overlap():
    assert cis_overlap((0.1, 0.3), (0.2, 0.4))
    assert not cis_overlap((0.1, 0.2), (0.3, 0.4))
    m = significance_matrix({"a": (0.1, 0.05, 0.15), "b": (0.5, 0.45, 0.55)})
    assert m["a"]["b"] and m["b"]["a"]
    assert not m["a"]["a"]



# --- backends --------------------------------------------------------------

def _fast_available():
    from tibetan_wer.alignment import available_backends
    return "fast" in available_backends()


def test_backends_agree_on_distance_and_rates():
    if not _fast_available():
        print("  (skipped: no fast backend installed)")
        return
    from tibetan_wer.alignment import distance
    pairs = [(REF, HYP), ("abcdef", "abdcef"), ("", "abc"), ("abc", ""), (REF, REF)]
    for ref, hyp in pairs:
        assert distance(list(ref), list(hyp), backend="python") == \
               distance(list(ref), list(hyp), backend="fast")
    slow = cer([HYP], [REF], backend="python")
    fast = cer([HYP], [REF], backend="fast")
    assert slow["micro_cer"] == fast["micro_cer"]
    assert slow["macro_cer"] == fast["macro_cer"]


def test_detail_rates_matches_detail_full_on_rates():
    full = ser([HYP, REF], [REF, HYP], detail="full")
    rates = ser([HYP, REF], [REF, HYP], detail="rates")
    assert full["micro_ser"] == rates["micro_ser"]
    assert full["macro_ser"] == rates["macro_ser"]
    assert np.allclose(full["per_utterance"], rates["per_utterance"])
    assert rates["substitutions"] is None       # the counts are what it gives up
    assert full["substitutions"] is not None


def test_auto_backend_keeps_the_reference_operations():
    from tibetan_wer.alignment import edit_counts
    a = edit_counts(list(REF), list(HYP))
    b = edit_counts(list(REF), list(HYP), backend="python")
    assert a == b


def test_unknown_backend_and_detail_raise():
    for bad in (lambda: cer([HYP], [REF], backend="turbo"),
                lambda: cer([HYP], [REF], detail="everything")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


# --- jiwer cross-check -----------------------------------------------------

def test_word_rate_matches_jiwer_on_space_delimited_text():
    try:
        import jiwer
    except ImportError:
        print("  (skipped: jiwer not installed)")
        return
    refs = ["the cat sat on the mat", "a second reference sentence here"]
    hyps = ["the cat sat on a mat", "second reference sentence"]
    theirs = jiwer.process_words(refs, hyps)
    ours = score_segments([h.split() for h in hyps], [r.split() for r in refs])
    assert abs(ours["micro_wer"] - theirs.wer) < 1e-12
    assert ours["substitutions"] == theirs.substitutions
    assert ours["insertions"] == theirs.insertions
    assert ours["deletions"] == theirs.deletions


# --- normalization ---------------------------------------------------------

def test_normalize_folds_variant_punctuation():
    from tibetan_wer import normalize
    assert normalize("ཀ༌ཁ") == "ཀ་ཁ"          # tsek bstar -> tsek
    assert normalize("ཀ༈ཁ") == "ཀ།ཁ"          # sbrul shad -> shad
    assert normalize("  ཀ་ཁ  ") == "ཀ་ཁ"
    assert normalize("ཀ༌ཁ", fold_tsek_bstar=False) == "ཀ༌ཁ"


def test_normalize_changes_ser_by_creating_a_boundary():
    plain = ser(["ཀ༌ཁ་ག"], ["ཀ་ཁ་ག"])
    folded = ser(["ཀ༌ཁ་ག"], ["ཀ་ཁ་ག"], normalize=True)
    assert plain["micro_ser"] > 0
    assert folded["micro_ser"] == 0.0


def test_normalize_accepts_kwargs():
    r = cer(["ཀ༌ཁ"], ["ཀ་ཁ"], normalize={"fold_tsek_bstar": False})
    assert r["micro_cer"] > 0


# --- segmentation fallbacks ------------------------------------------------

def test_fallbacks_are_counted_and_reported():
    import tibetan_wer.segmentation as seg
    from tibetan_wer import fallback_counts, reset_fallback_counts

    class Boom:
        def tokenize(self, text):
            raise RuntimeError("botok blew up")

    reset_fallback_counts()
    tokens = seg.word_segment("ཀ་ཁ་", tokenizer=Boom())
    assert tokens == ["ཀ", "ཁ"]                  # fell back to syllables
    assert fallback_counts()["botok"] == 1
    assert reset_fallback_counts()["botok"] == 1
    assert fallback_counts() == {}


def test_on_error_raise_propagates():
    import tibetan_wer.segmentation as seg

    class Boom:
        def tokenize(self, text):
            raise RuntimeError("botok blew up")

    try:
        seg.word_segment("ཀ་ཁ་", tokenizer=Boom(), on_error="raise")
    except RuntimeError:
        return
    raise AssertionError("expected the error to propagate")


def test_metrics_report_fallback_count():
    import tibetan_wer.segmentation as seg
    from tibetan_wer import reset_fallback_counts, wer
    if not _botok_available():
        print("  (skipped: botok not installed)")
        return
    reset_fallback_counts()
    r = wer([HYP], [REF])
    assert r["num_segmentation_fallbacks"] == 0


# --- paired comparison -----------------------------------------------------

def _two_systems():
    refs = ["abcdefgh"] * 40
    good = ["abcdefgh"] * 36 + ["abcdefgX"] * 4
    bad = ["abcdefgX"] * 40
    return good, bad, refs


def test_compare_detects_a_difference_and_signs_it():
    good, bad, refs = _two_systems()
    r = compare(cer(good, refs), cer(bad, refs), n_iterations=2000)
    assert r["delta"] < 0                        # a is better than b
    assert r["ci"][1] < 0 and r["significant"]
    assert r["p_permutation"] < 0.05
    assert r["n_paired"] == 40 and r["n_dropped"] == 0


def test_compare_finds_nothing_between_identical_systems():
    good, _, refs = _two_systems()
    r = compare(cer(good, refs), cer(good, refs), n_iterations=1000)
    assert r["delta"] == 0.0
    assert not r["significant"]
    assert r["p_permutation"] == 1.0


def test_compare_is_paired_and_drops_unscorable_lines():
    r = compare(cer(["ab", "xy"], ["ab", ""]), cer(["ab", "zz"], ["ab", ""]), n_iterations=500)
    assert r["n_paired"] == 1 and r["n_dropped"] == 1


def test_compare_micro_needs_the_result_dict():
    good, bad, refs = _two_systems()
    a, b = cer(good, refs), cer(bad, refs)
    micro = compare(a, b, n_iterations=500, statistic="micro")
    assert micro["rate_a"] == a["micro_cer"]
    try:
        compare(a["per_utterance"], b["per_utterance"], statistic="micro")
    except ValueError:
        return
    raise AssertionError("expected ValueError without per-sentence counts")


def test_compare_rejects_mismatched_lengths():
    try:
        compare(cer(["a"], ["a"]), cer(["a", "b"], ["a", "b"]))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_bootstrap_micro_matches_the_reported_micro_rate():
    good, _, refs = _two_systems()
    r = cer(good, refs)
    point, lo, hi = bootstrap_ci(r, n_iterations=1000, statistic="micro")
    assert abs(point - r["micro_cer"]) < 1e-12
    assert lo <= point <= hi


# --- error profile ---------------------------------------------------------

def test_profile_counts_operations_and_shares():
    from tibetan_wer import error_profile
    p = error_profile(["abd"], ["abc"], level="char")
    assert p["operations"]["substitutions"] == 1
    assert p["operations"]["substitution_share"] == 1.0
    assert p["confusions"][0] == {"reference": "c", "hypothesis": "d", "count": 1}


def test_profile_counts_boundary_and_space_edits():
    from tibetan_wer import error_profile
    p = error_profile(["ཀ ཁ"], ["ཀ་ཁ"], level="char")   # tsek replaced by a space
    assert p["boundary_edits"]["count"] == 1
    assert p["space_edits"]["count"] == 1
    assert p["space_edits"]["share"] == 1.0


def test_profile_malformed_share_needs_a_vocabulary():
    from tibetan_wer import error_profile
    vocab = {"ཀ", "ཁ", "ག"}
    p = error_profile(["ཀ་ག་"], ["ཀ་ཁ་"], level="syllable", vocabulary=vocab)
    assert p["malformed"]["attested"] == 1 and p["malformed"]["malformed"] == 0
    q = error_profile(["ཀ་ཟ་"], ["ཀ་ཁ་"], level="syllable", vocabulary=vocab)
    assert q["malformed"]["malformed"] == 1 and q["malformed"]["share"] == 1.0
    assert "malformed" not in error_profile(["ཀ་ཟ་"], ["ཀ་ཁ་"], level="syllable")


def test_concentration_and_gini():
    from tibetan_wer import concentration, gini
    even = concentration([0.5, 0.5, 0.5, 0.5])
    assert even["zero_error_share"] == 0.0
    assert abs(even["gini"]) < 1e-9
    skewed = concentration([0.0] * 9 + [1.0])
    assert skewed["zero_error_share"] == 0.9
    assert skewed["worst_decile_share"] == 1.0
    assert gini([1.0]) != gini([1.0])            # nan with fewer than two nonzero values


# --- CLI -------------------------------------------------------------------

def test_cli_reports_rates_and_a_comparison(tmp_path=None):
    import json as _json
    import tempfile
    from tibetan_wer.cli import main

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "refs.txt").write_text("ཀ་ཁ་ག\nང་ཅ་ཆ", encoding="utf-8")
        (base / "a.txt").write_text("ཀ་ཁ་ག\nང་ཅ་ཇ", encoding="utf-8")
        (base / "b.txt").write_text("ཀ་ཁ་ཟ\nཟ་ཟ་ཟ", encoding="utf-8")
        out = base / "out.json"
        code = main([str(base / "a.txt"), str(base / "refs.txt"), "--metric", "ser",
                     "--ci", "--iterations", "200", "--compare", str(base / "b.txt"),
                     "--profile", "--json", str(out)])
        assert code == 0
        report = _json.loads(out.read_text(encoding="utf-8"))
        assert report["metric"] == "ser" and report["num_sentences"] == 2
        assert report["comparison"]["delta"] < 0        # a beats b
        assert "profile" in report and "ci" in report


def test_cli_rejects_mismatched_files():
    import tempfile
    from tibetan_wer.cli import main

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "refs.txt").write_text("ཀ་ཁ\nག་ང", encoding="utf-8")
        (base / "a.txt").write_text("ཀ་ཁ", encoding="utf-8")
        assert main([str(base / "a.txt"), str(base / "refs.txt")]) == 2


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)

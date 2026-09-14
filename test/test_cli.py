"""Tests for the command-line interface.

Covers the text report, the JSON report, every ``--metric`` branch (the two
that would call out to a model or an API are dispatched to a stub), the
vocabulary options, stdin input, and the exit statuses.

Runs under pytest, or standalone with `python test/test_cli.py`.
"""
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tibetan_wer.metrics as M  # noqa: E402
from tibetan_wer.cli import build_parser, main  # noqa: E402

REFS = "ཀ་ཁ་ག\nང་ཅ་ཆ\nཇ་ཉ་ཏ"
GOOD = "ཀ་ཁ་ག\nང་ཅ་ཇ\nཇ་ཉ་ཏ"
POOR = "ཀ་ཁ་ཟ\nཟ་ཟ་ཟ\nཇ་ཟ་ཟ"


class _Files:
    def __init__(self, stack):
        self.dir = Path(stack)
        (self.dir / "refs.txt").write_text(REFS, encoding="utf-8")
        (self.dir / "good.txt").write_text(GOOD, encoding="utf-8")
        (self.dir / "poor.txt").write_text(POOR, encoding="utf-8")

    def __getattr__(self, name):
        return str(self.dir / (name + ".txt"))


def _run(args):
    """Run the CLI, returning (exit status, stdout)."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(args)
    return code, buffer.getvalue()


def _with_files(fn):
    with tempfile.TemporaryDirectory() as tmp:
        return fn(_Files(tmp))


# --- text report -----------------------------------------------------------

def test_text_report_shows_rates_and_counts():
    def check(f):
        code, out = _run([f.good, f.refs, "--metric", "ser"])
        assert code == 0
        assert "SER over 3 sentences" in out
        assert "micro" in out and "macro" in out
        assert "S/I/D" in out
    _with_files(check)


def test_text_report_shows_an_interval():
    def check(f):
        _, out = _run([f.good, f.refs, "--ci", "--iterations", "200"])
        assert "95% CI" in out and "(macro rate, bootstrap)" in out
        _, micro = _run([f.good, f.refs, "--ci", "--iterations", "200", "--statistic", "micro"])
        assert "(micro rate, bootstrap)" in micro
    _with_files(check)


def test_text_report_shows_a_comparison():
    def check(f):
        _, out = _run([f.good, f.refs, "--compare", f.poor, "--iterations", "200"])
        assert "comparison (macro rate, paired over 3 sentences)" in out
        assert "delta -" in out                      # the good system wins
        assert "p (bootstrap)" in out and "p (permutation)" in out
        assert "significant" in out
    _with_files(check)


def test_text_report_shows_a_profile_with_a_vocabulary_file():
    def check(f):
        (f.dir / "vocab.txt").write_text("ཀ\nཁ\nག\nང\nཅ\nཆ\nཇ\nཉ\nཏ", encoding="utf-8")
        _, out = _run([f.poor, f.refs, "--profile", "--level", "syllable",
                       "--vocabulary", str(f.dir / "vocab.txt")])
        assert "error profile (syllable level" in out
        assert "malformed syllables produced" in out
        assert "clean lines" in out and "gini" in out
    _with_files(check)


def test_vocabulary_dash_builds_one_from_the_references():
    def check(f):
        _, out = _run([f.poor, f.refs, "--profile", "--level", "syllable", "--vocabulary", "-"])
        assert "vocabulary 9" in out                 # nine distinct syllables in REFS
    _with_files(check)


def test_char_profile_reports_boundary_and_space_shares():
    def check(f):
        _, out = _run([f.poor, f.refs, "--profile"])
        assert "error profile (char level" in out
        assert "on the tsek" in out and "producing a space" in out
        assert "top confusions:" in out
    _with_files(check)


# --- JSON report -----------------------------------------------------------

def test_json_to_stdout_is_parseable():
    def check(f):
        _, out = _run([f.good, f.refs, "--json"])
        report = json.loads(out)
        assert report["metric"] == "ser" and report["num_sentences"] == 3
        assert "per_sentence" not in report          # dropped: too bulky for a report
    _with_files(check)


def test_json_to_a_file_holds_the_same_report():
    def check(f):
        out_path = f.dir / "out.json"
        code, _ = _run([f.good, f.refs, "--compare", f.poor, "--ci",
                        "--iterations", "200", "--json", str(out_path)])
        assert code == 0
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["comparison"]["delta"] < 0
        assert report["ci"]["lo"] <= report["ci"]["point"] <= report["ci"]["hi"]
    _with_files(check)


# --- options ---------------------------------------------------------------

def test_every_metric_branch_dispatches():
    calls = []

    def stub(name):
        def fn(predictions, references, **kwargs):
            calls.append((name, kwargs))
            units = [r.split("་") for r in references]
            return M.score_segments([p.split("་") for p in predictions], units)
        return fn

    real_bert, real_gemini = M.bert_wer, M.gemini_wer
    M.bert_wer, M.gemini_wer = stub("bert"), stub("gemini")
    try:
        def check(f):
            for metric in ("cer", "ser", "wer", "bert", "gemini"):
                code, out = _run([f.good, f.refs, "--metric", metric])
                assert code == 0 and out.strip()
        _with_files(check)
    finally:
        M.bert_wer, M.gemini_wer = real_bert, real_gemini
    assert [name for name, _ in calls] == ["bert", "gemini"]
    assert calls[1][1]["workers"] == 1               # --workers reaches gemini_wer


def test_normalize_and_backend_and_detail_flags_take_effect():
    def check(f):
        (f.dir / "variant.txt").write_text("ཀ༌ཁ་ག\nང་ཅ་ཆ\nཇ་ཉ་ཏ", encoding="utf-8")
        plain = json.loads(_run([str(f.dir / "variant.txt"), f.refs, "--json"])[1])
        folded = json.loads(_run([str(f.dir / "variant.txt"), f.refs, "--normalize", "--json"])[1])
        assert plain["micro"] > 0 and folded["micro"] == 0.0
        rates = json.loads(_run([f.good, f.refs, "--detail", "rates", "--json"])[1])
        assert "substitutions" not in rates          # not computed in this mode
        fast = json.loads(_run([f.good, f.refs, "--backend", "python", "--json"])[1])
        assert fast["substitutions"] is not None
    _with_files(check)


def test_predictions_can_come_from_stdin():
    def check(f):
        stdin = sys.stdin
        sys.stdin = io.StringIO(GOOD)
        try:
            code, out = _run(["-", f.refs])
        finally:
            sys.stdin = stdin
        assert code == 0 and "SER over 3 sentences" in out
    _with_files(check)


def test_mismatched_line_counts_exit_two():
    def check(f):
        (f.dir / "short.txt").write_text("ཀ་ཁ་ག", encoding="utf-8")
        code, _ = _run([str(f.dir / "short.txt"), f.refs])
        assert code == 2
    _with_files(check)


def test_report_mentions_skipped_sentences_and_fallbacks():
    def stub(predictions, references, **kwargs):
        units = [r.split("་") for r in references]
        result = M.score_segments([p.split("་") for p in predictions], units)
        result["num_skipped"] = 2
        result["num_segmentation_fallbacks"] = 3
        return result

    real = M.gemini_wer
    M.gemini_wer = stub
    try:
        def check(f):
            _, out = _run([f.good, f.refs, "--metric", "gemini"])
            assert "skipped 2 sentences" in out
            assert "fell back to syllables on 3 strings" in out
        _with_files(check)
    finally:
        M.gemini_wer = real


def test_parser_defaults():
    args = build_parser().parse_args(["a.txt", "b.txt"])
    assert args.metric == "ser" and args.detail == "full" and args.backend == "auto"
    assert args.statistic == "macro" and args.iterations == 10_000 and args.seed == 42
    assert args.json is None and args.workers == 1


def test_module_entry_point_runs():
    import subprocess

    def check(f):
        out = subprocess.run(
            [sys.executable, "-m", "tibetan_wer", f.good, f.refs, "--metric", "cer"],
            capture_output=True, text=True,
            env={"PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
                 "PATH": "/usr/bin:/bin"},
        )
        assert out.returncode == 0, out.stderr
        assert "CER over 3 sentences" in out.stdout
    _with_files(check)


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

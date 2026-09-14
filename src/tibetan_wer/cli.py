"""Command-line interface.

    tibetan-wer predictions.txt references.txt --metric ser --ci
    tibetan-wer a.txt refs.txt --compare b.txt --metric wer
    tibetan-wer preds.txt refs.txt --profile --level syllable --json out.json

Files hold one sentence per line, UTF-8, prediction *i* against reference *i*.
"""
from __future__ import annotations

import argparse
import json
import sys

METRICS = ("cer", "ser", "wer", "bert", "gemini")


def _read(path):
    if path == "-":
        return [line.rstrip("\n") for line in sys.stdin]
    with open(path, encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _score(predictions, references, metric, args):
    from . import metrics as M

    common = {"normalize": args.normalize, "detail": args.detail, "backend": args.backend}
    if metric == "cer":
        return M.cer(predictions, references, **common), "cer"
    if metric == "ser":
        return M.ser(predictions, references, **common), "ser"
    if metric == "wer":
        return M.wer(predictions, references, **common), "wer"
    if metric == "bert":
        return M.bert_wer(predictions, references, **common), "wer"
    if metric == "gemini":
        return M.gemini_wer(predictions, references, workers=args.workers, **common), "wer"
    raise ValueError("unknown metric {!r}".format(metric))


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "per_sentence"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [None if np.isnan(v) else float(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def build_parser() -> argparse.ArgumentParser:
    """The argument parser, exposed for documentation and shell completion."""
    parser = argparse.ArgumentParser(
        prog="tibetan-wer",
        description="Character, syllable and segmented word error rates for Tibetan text.",
    )
    parser.add_argument("predictions", help="file of predicted lines, or - for stdin")
    parser.add_argument("references", help="file of reference lines")
    parser.add_argument("--metric", default="ser", choices=METRICS, help="default: ser")
    parser.add_argument("--compare", metavar="FILE", help="second system, paired against the first")
    parser.add_argument("--ci", action="store_true", help="bootstrap confidence interval")
    parser.add_argument("--statistic", default="macro", choices=("macro", "micro"))
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true", help="NFC, fold tsek bstar and sbrul shad, strip")
    parser.add_argument("--detail", default="full", choices=("full", "rates"),
                        help="'rates' skips edit-operation counts and runs far faster")
    parser.add_argument("--backend", default="auto", choices=("auto", "python", "fast"))
    parser.add_argument("--workers", type=int, default=1, help="concurrent API calls (gemini)")
    parser.add_argument("--profile", action="store_true", help="report error-structure diagnostics")
    parser.add_argument("--level", default="char", choices=("char", "syllable", "botok", "bert", "gemini"),
                        help="unit for --profile")
    parser.add_argument("--vocabulary", metavar="FILE",
                        help="attested syllables for --profile --level syllable; "
                             "'-' builds one from the references")
    parser.add_argument("--json", metavar="FILE", nargs="?", const="-",
                        help="write JSON instead of text (default: stdout)")
    return parser


def main(argv=None) -> int:
    """Run the command line. Returns a process exit status.

    ``argv`` defaults to ``sys.argv[1:]``; pass a list to call it in-process.
    Exit 0 on success, 2 when the two files disagree on how many lines they
    hold.
    """
    args = build_parser().parse_args(argv)
    predictions = _read(args.predictions)
    references = _read(args.references)
    if len(predictions) != len(references):
        print(
            "error: {} predictions but {} references".format(len(predictions), len(references)),
            file=sys.stderr,
        )
        return 2

    report = {"metric": args.metric, "num_sentences": len(references)}
    result, unit = _score(predictions, references, args.metric, args)
    report["micro"] = result["micro_" + unit]
    report["macro"] = result["macro_" + unit]
    for key in ("substitutions", "insertions", "deletions", "num_scored",
                "num_skipped", "num_segmentation_fallbacks"):
        if result.get(key) is not None:
            report[key] = result[key]

    if args.ci:
        from .stats import bootstrap_ci

        point, lo, hi = bootstrap_ci(
            result, n_iterations=args.iterations, seed=args.seed, statistic=args.statistic
        )
        report["ci"] = {"statistic": args.statistic, "point": point, "lo": lo, "hi": hi}

    if args.compare:
        from .stats import compare

        other, _ = _score(_read(args.compare), references, args.metric, args)
        report["comparison"] = compare(
            result, other, n_iterations=args.iterations, seed=args.seed, statistic=args.statistic
        )
        report["comparison"]["a"] = args.predictions
        report["comparison"]["b"] = args.compare

    if args.profile:
        from .profile import error_profile
        from .segmentation import syllable_segment

        vocabulary = None
        if args.vocabulary == "-":
            vocabulary = {s for r in references for s in syllable_segment(r)}
        elif args.vocabulary:
            vocabulary = {s.strip() for s in _read(args.vocabulary) if s.strip()}
        report["profile"] = error_profile(
            predictions, references, level=args.level,
            vocabulary=vocabulary, normalize=args.normalize,
        )

    if args.json:
        payload = json.dumps(_jsonable(report), ensure_ascii=False, indent=2)
        if args.json == "-":
            print(payload)
        else:
            with open(args.json, "w", encoding="utf-8") as handle:
                handle.write(payload + "\n")
        return 0

    _print_text(report, unit)
    return 0


def _print_text(report, unit):
    name = {"cer": "CER", "ser": "SER", "wer": "WER"}[unit]
    print("{} over {} sentences".format(name, report["num_sentences"]))
    print("  micro  {:.4f}".format(report["micro"]))
    print("  macro  {:.4f}".format(report["macro"]))
    if "ci" in report:
        ci = report["ci"]
        print("  95% CI [{:.4f}, {:.4f}]  ({} rate, bootstrap)".format(
            ci["lo"], ci["hi"], ci["statistic"]))
    if report.get("substitutions") is not None:
        print("  S/I/D  {} / {} / {}".format(
            report["substitutions"], report["insertions"], report["deletions"]))
    if report.get("num_skipped"):
        print("  skipped {} sentences".format(report["num_skipped"]))
    if report.get("num_segmentation_fallbacks"):
        print("  segmentation fell back to syllables on {} strings".format(
            report["num_segmentation_fallbacks"]))
    if "comparison" in report:
        c = report["comparison"]
        print("\ncomparison ({} rate, paired over {} sentences)".format(c["statistic"], c["n_paired"]))
        print("  a  {}  {:.4f}".format(c["a"], c["rate_a"]))
        print("  b  {}  {:.4f}".format(c["b"], c["rate_b"]))
        print("  delta {:+.4f}  95% CI [{:+.4f}, {:+.4f}]".format(c["delta"], c["ci"][0], c["ci"][1]))
        print("  p (bootstrap) {:.4f}   p (permutation) {:.4f}".format(
            c["p_bootstrap"], c["p_permutation"]))
        print("  {} at the 5% level".format(
            "significant" if c["significant"] else "not significant"))
    if "profile" in report:
        p = report["profile"]
        ops = p["operations"]
        print("\nerror profile ({} level, {} edits)".format(p["level"], ops["total"]))
        print("  substitutions {:.1%}   insertions {:.1%}   deletions {:.1%}".format(
            ops["substitution_share"], ops["insertion_share"], ops["deletion_share"]))
        if "boundary_edits" in p:
            print("  on the tsek   {:.1%}   producing a space {:.1%}".format(
                p["boundary_edits"]["share"], p["space_edits"]["share"]))
        if "malformed" in p:
            print("  malformed syllables produced {:.1%} (vocabulary {:,})".format(
                p["malformed"]["share"], p["malformed"]["vocabulary_size"]))
        conc = p["concentration"]
        print("  clean lines {:.1%}   p90 {:.3f}   worst decile holds {:.1%}   gini {:.3f}".format(
            conc["zero_error_share"], conc["p90"], conc["worst_decile_share"], conc["gini"]))
        if p["confusions"]:
            top = ", ".join("{}->{} ({})".format(c["reference"], c["hypothesis"], c["count"])
                            for c in p["confusions"][:5])
            print("  top confusions: " + top)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Error rates for Tibetan text.

Metrics       cer, ser, wer / botok_wer, bert_wer, gemini_wer, score_segments
Statistics    bootstrap_ci, compare, cis_overlap, significance_matrix
Diagnostics   error_profile, concentration, gini, edit_operations
Alignment     align, edit_counts, distance, EditOperation, available_backends
Segmentation  syllable_segment, word_segment, bert_segment, gemini_segment,
              segment_all, fallback_counts, repetition_ratio, is_degenerate
Normalization normalize, normalize_all

A command-line entry point is installed as ``tibetan-wer``; see ``--help``.
"""
from .alignment import EditOperation, align, available_backends, distance, edit_counts
from .metrics import (
    bert_wer,
    botok_wer,
    cer,
    edit_operations,
    gemini_wer,
    score_segments,
    ser,
    wer,
)
from .normalization import normalize, normalize_all
from .profile import concentration, error_profile, gini
from .segmentation import (
    bert_segment,
    fallback_counts,
    gemini_segment,
    is_degenerate,
    repetition_ratio,
    reset_fallback_counts,
    segment_all,
    syllable_segment,
    word_segment,
)
from .stats import bootstrap_ci, cis_overlap, compare, significance_matrix

__version__ = "1.2.0"

__all__ = [
    # metrics
    "cer",
    "ser",
    "wer",
    "botok_wer",
    "bert_wer",
    "gemini_wer",
    "score_segments",
    # statistics
    "bootstrap_ci",
    "compare",
    "cis_overlap",
    "significance_matrix",
    # diagnostics
    "error_profile",
    "concentration",
    "gini",
    "edit_operations",
    # alignment
    "align",
    "edit_counts",
    "distance",
    "EditOperation",
    "available_backends",
    # segmentation
    "syllable_segment",
    "word_segment",
    "bert_segment",
    "gemini_segment",
    "segment_all",
    "fallback_counts",
    "reset_fallback_counts",
    "repetition_ratio",
    "is_degenerate",
    # normalization
    "normalize",
    "normalize_all",
    "__version__",
]

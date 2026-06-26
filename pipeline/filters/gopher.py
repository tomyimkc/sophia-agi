# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gopher (MassiveText) quality + repetition heuristics — pure stdlib.

Rae et al. 2021, Appendix A. The widely-reused document-level quality rules; a
document is kept only if it passes ALL registered thresholds (fail-closed). Each
signal is computed deterministically so the keep decision is reproducible.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"\S+")
_ALPHA_RE = re.compile(r"[A-Za-z]")
# Common English stop words used by the Gopher "must contain ≥2 stop words" rule.
STOP_WORDS = frozenset(
    "the be to of and that have with this from not are was you it for on as".split()
)

# Pre-registered thresholds (Rae et al. 2021).
THRESHOLDS = {
    "min_words": 50,
    "max_words": 100_000,
    "min_mean_word_len": 3.0,
    "max_mean_word_len": 10.0,
    "max_symbol_to_word": 0.10,          # '#' and ellipsis
    "max_bullet_line_frac": 0.90,
    "max_ellipsis_line_frac": 0.30,
    "min_alpha_word_frac": 0.80,         # ≥80% of words contain an alpha char
    "min_stop_words": 2,
    "max_dup_line_frac": 0.30,
}


def _words(text: str) -> "list[str]":
    return _WORD_RE.findall(text)


def signals(text: str) -> dict:
    """Compute the Gopher signals for a document (deterministic, pure stdlib)."""
    words = _words(text)
    n = len(words)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    n_lines = len(lines) or 1
    n_sym = text.count("#") + text.count("…") + text.count("...")
    bullet = sum(1 for ln in lines if ln.lstrip()[:1] in {"•", "-", "*", "·"})
    ellipsis = sum(1 for ln in lines if ln.rstrip().endswith(("…", "...")))
    alpha_words = sum(1 for w in words if _ALPHA_RE.search(w))
    stop = sum(1 for w in words if w.lower() in STOP_WORDS)
    seen: dict[str, int] = {}
    for ln in lines:
        seen[ln] = seen.get(ln, 0) + 1
    dup_lines = sum(c for c in seen.values() if c > 1) - sum(1 for c in seen.values() if c > 1)
    return {
        "wordCount": n,
        "meanWordLen": (sum(len(w) for w in words) / n) if n else 0.0,
        "symbolToWordRatio": (n_sym / n) if n else 0.0,
        "bulletLineFrac": bullet / n_lines,
        "ellipsisLineFrac": ellipsis / n_lines,
        "alphaWordFrac": (alpha_words / n) if n else 0.0,
        "stopWordCount": stop,
        "dupLineFrac": dup_lines / n_lines,
    }


def keep(text: str) -> "tuple[bool, list[str]]":
    """Fail-closed Gopher keep decision. Returns (keep, list-of-failed-rules)."""
    s = signals(text)
    T = THRESHOLDS
    fails = []
    if not (T["min_words"] <= s["wordCount"] <= T["max_words"]):
        fails.append("word_count")
    if not (T["min_mean_word_len"] <= s["meanWordLen"] <= T["max_mean_word_len"]):
        fails.append("mean_word_len")
    if s["symbolToWordRatio"] > T["max_symbol_to_word"]:
        fails.append("symbol_to_word")
    if s["bulletLineFrac"] > T["max_bullet_line_frac"]:
        fails.append("bullet_lines")
    if s["ellipsisLineFrac"] > T["max_ellipsis_line_frac"]:
        fails.append("ellipsis_lines")
    if s["alphaWordFrac"] < T["min_alpha_word_frac"]:
        fails.append("alpha_word_frac")
    if s["stopWordCount"] < T["min_stop_words"]:
        fails.append("stop_words")
    if s["dupLineFrac"] > T["max_dup_line_frac"]:
        fails.append("dup_lines")
    return (not fails), fails

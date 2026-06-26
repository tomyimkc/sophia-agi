# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""C4 line-level cleaning heuristics — pure stdlib.

Raffel et al. 2020 (T5/C4), §2.2. Line-level rules that strip boilerplate/code
and drop low-quality pages. Defensive subset: we keep the structural/quality rules
(terminal punctuation, min words/sentences, code-brace and placeholder removal) and
deliberately omit any "bad-words" blocklist so no hazardous content is enumerated.
"""
from __future__ import annotations

_TERMINAL = (".", "!", "?", '"', "”")
_PLACEHOLDER = "lorem ipsum"


def clean_lines(text: str) -> "tuple[str, dict]":
    """Apply C4 line filters; return (cleaned_text, signals).

    Drops: lines not ending in terminal punctuation; lines with < 3 words; lines
    containing a curly brace (JavaScript/code) or the lorem-ipsum placeholder.
    """
    lines = text.split("\n")
    kept = []
    dropped = 0
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        low = s.lower()
        if "{" in s or "}" in s:           # code / JSON / JS boilerplate
            dropped += 1
            continue
        if _PLACEHOLDER in low:
            dropped += 1
            continue
        if len(s.split()) < 3:
            dropped += 1
            continue
        if not s.endswith(_TERMINAL):
            dropped += 1
            continue
        kept.append(s)
    cleaned = "\n".join(kept)
    return cleaned, {
        "linesIn": sum(1 for ln in lines if ln.strip()),
        "linesKept": len(kept),
        "linesDropped": dropped,
    }


def _sentence_count(text: str) -> int:
    return sum(text.count(p) for p in (".", "!", "?"))


def keep(text: str, *, min_sentences: int = 5) -> "tuple[bool, list[str], str]":
    """Fail-closed C4 page decision. Returns (keep, failed-rules, cleaned_text).

    A page is dropped if, after line cleaning, it has fewer than `min_sentences`
    sentences or no surviving lines.
    """
    cleaned, sig = clean_lines(text)
    fails = []
    if sig["linesKept"] == 0:
        fails.append("no_lines_survived")
    if _sentence_count(cleaned) < min_sentences:
        fails.append("too_few_sentences")
    return (not fails), fails, cleaned

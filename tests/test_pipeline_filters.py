# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Data-platform M0 — Gopher/C4 quality filter tests (plain-script, stdlib)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.filters import c4, gopher, quality  # noqa: E402

# A clean, prose-like document that should pass both filters.
GOOD = (
    "The history of writing is long and varied. Scribes recorded harvests and laws "
    "on clay tablets. Later, paper made books cheaper to produce and to share. "
    "Printing spread ideas across the world at a pace never seen before. Today, "
    "digital text reaches billions of readers within seconds of being written.\n"
    "Many scholars study how these changes shaped human thought and memory. "
    "They argue that each new medium altered what people chose to record."
)
# Code/boilerplate spam: curly braces, short lines, no terminal punctuation.
SPAM = "function f() {\nreturn x\n}\nclick here\nbuy now\n{json: true}\nlorem ipsum dolor"
SHORT = "Too short."


def test_gopher_signals_basic() -> None:
    s = gopher.signals(GOOD)
    assert s["wordCount"] > 50
    assert 3.0 <= s["meanWordLen"] <= 10.0
    assert s["stopWordCount"] >= 2
    assert s["alphaWordFrac"] > 0.8


def test_gopher_keeps_good_drops_garbage() -> None:
    ok, fails = gopher.keep(GOOD)
    assert ok and fails == []
    bad = "# # # ... ... ###\n• x\n• y\n• z"  # symbol/bullet heavy, too short
    ok2, fails2 = gopher.keep(bad)
    assert not ok2 and fails2


def test_c4_clean_lines_drops_code_and_short() -> None:
    cleaned, sig = c4.clean_lines(SPAM)
    assert "{" not in cleaned and "lorem ipsum" not in cleaned.lower()
    assert sig["linesDropped"] >= 1
    # Every surviving line ends in terminal punctuation.
    for ln in cleaned.split("\n"):
        if ln:
            assert ln.endswith((".", "!", "?", '"', "”"))


def test_quality_tagger_keep_decisions() -> None:
    assert quality.tag_document(GOOD)["keep"] is True
    assert quality.tag_document(SPAM)["keep"] is False
    assert quality.tag_document(SHORT)["keep"] is False


def test_filter_corpus_stats_and_determinism() -> None:
    docs = [GOOD, SPAM, SHORT, GOOD]
    r = quality.filter_corpus(docs)
    assert r["n"] == 4 and r["kept"] == 2 and r["dropped"] == 2
    assert abs(r["keepRate"] - 0.5) < 1e-9
    assert r["dropReasons"]  # non-empty histogram
    assert quality.filter_corpus(docs) == r  # deterministic


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"PASSED {len(tests)} pipeline-filter tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())

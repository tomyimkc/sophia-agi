#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the context-window manager (deterministic, offline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import context_manager as cm  # noqa: E402


def test_estimate_tokens_bilingual_and_monotonic() -> None:
    assert cm.estimate_tokens("") == 0
    assert cm.estimate_tokens("x") >= 1
    # CJK bills heavier per char than Latin: 8 hanzi outweigh 8 ascii chars.
    assert cm.estimate_tokens("中文摘要测试一下") > cm.estimate_tokens("abcdefgh")
    # Monotonic in length.
    assert cm.estimate_tokens("a" * 400) > cm.estimate_tokens("a" * 40)


def test_small_input_passes_through_unchanged() -> None:
    segs = [cm.Segment(kind="prior", text="short answer", provenance="p0")]
    out = cm.ContextManager(budget_tokens=10_000).pack(segs)
    assert out.text == "short answer"
    assert out.dropped == [] and out.compressed == []
    assert out.over_budget is False


def test_priority_drop_under_budget() -> None:
    # Budget only fits the high-priority segment; the low-priority one is dropped.
    big = "word " * 200  # ~250 tokens
    segs = [
        cm.Segment(kind="prior", text=big, priority=1, provenance="low"),
        cm.Segment(kind="prior", text=big, priority=9, provenance="high"),
    ]
    # Budget fits the high-priority seg (~250 tok) with too little left to compress
    # the low one above the floor => it is dropped.
    out = cm.ContextManager(budget_tokens=260, compress_floor_tokens=50).pack(segs)
    assert "high" in out.kept
    assert "low" in out.dropped
    assert out.tokens <= out.budget


def test_pinned_is_never_dropped() -> None:
    big = "word " * 200
    segs = [
        cm.Segment(kind="system", text=big, pinned=True, provenance="sys"),
        cm.Segment(kind="prior", text=big, priority=9, provenance="other"),
    ]
    out = cm.ContextManager(budget_tokens=120).pack(segs)
    # Pinned survives (kept or compressed), never dropped.
    assert "sys" not in out.dropped
    assert "sys" in out.kept or "sys" in out.compressed


def test_pinned_overflow_is_flagged_not_silently_dropped() -> None:
    huge = "word " * 5000
    segs = [cm.Segment(kind="system", text=huge, pinned=True, stable=True, provenance="sys")]
    out = cm.ContextManager(budget_tokens=100).pack(segs)
    assert out.over_budget is True
    assert "sys" not in out.dropped  # fail-closed: kept, just flagged


def test_stable_prefix_cache_key_is_stable_across_volatile_changes() -> None:
    stable_seg = cm.Segment(kind="system", text="SYSTEM PROMPT", stable=True, provenance="sys")
    a = cm.ContextManager(10_000).pack([stable_seg, cm.Segment(kind="prior", text="turn 1", provenance="t")])
    b = cm.ContextManager(10_000).pack([stable_seg, cm.Segment(kind="prior", text="turn 2 differs", provenance="t")])
    # Volatile tail changed; the cache-stable prefix key must not.
    assert a.cache_key == b.cache_key
    # And the stable prefix is emitted first.
    assert a.text.startswith("SYSTEM PROMPT")
    # A change to the stable segment DOES change the key.
    c = cm.ContextManager(10_000).pack([cm.Segment(kind="system", text="DIFFERENT", stable=True, provenance="sys")])
    assert c.cache_key != a.cache_key


def test_head_tail_compress_keeps_both_ends_and_fits() -> None:
    lines = [f"line {i} content here" for i in range(200)]
    lines[-1] = "Decision: proceed. 中文摘要: 结论"
    text = "\n".join(lines)
    out = cm.head_tail_compress(text, budget=60, counter=cm.estimate_tokens)
    assert cm.estimate_tokens(out) <= 60 + 12  # within budget + marker slack
    assert "line 0 content here" in out  # head preserved
    assert "Decision: proceed" in out  # tail (Decision block) preserved
    assert "elided" in out  # explicit elision marker


def test_compress_floor_drops_rather_than_overcompress() -> None:
    big = "word " * 200
    segs = [
        cm.Segment(kind="prior", text="A" * 4, priority=9, provenance="keep"),
        cm.Segment(kind="prior", text=big, priority=1, provenance="floor"),
    ]
    # Tiny remaining budget after the high-priority seg => below compress floor => drop.
    out = cm.ContextManager(budget_tokens=8, compress_floor_tokens=50).pack(segs)
    assert "floor" in out.dropped


def test_compact_history_pins_most_recent() -> None:
    outputs = ["old " * 300, "middle " * 300, "RECENT decision 中文摘要"]
    text, pack = cm.compact_history(outputs, budget_tokens=120, keep_recent=1)
    # Most recent output is pinned and present verbatim.
    assert "RECENT decision" in text
    assert "prior#2" not in pack.dropped
    # Budget honoured (recent pinned may push over, but older must yield first).
    assert pack.dropped or pack.compressed


def test_summarizer_overshoot_falls_back_to_head_tail() -> None:
    def bad_summarizer(text: str, budget: int) -> str:
        return text  # ignores budget — manager must not trust it

    big = "\n".join(f"line {i}" for i in range(300))
    out = cm.ContextManager(budget_tokens=50, summarizer=bad_summarizer).pack(
        [cm.Segment(kind="prior", text=big, priority=1, provenance="s")]
    )
    assert out.tokens <= out.budget + 12  # fell back, fits within marker slack


def main() -> int:
    test_estimate_tokens_bilingual_and_monotonic()
    test_small_input_passes_through_unchanged()
    test_priority_drop_under_budget()
    test_pinned_is_never_dropped()
    test_pinned_overflow_is_flagged_not_silently_dropped()
    test_stable_prefix_cache_key_is_stable_across_volatile_changes()
    test_head_tail_compress_keeps_both_ends_and_fits()
    test_compress_floor_drops_rather_than_overcompress()
    test_compact_history_pins_most_recent()
    test_summarizer_overshoot_falls_back_to_head_tail()
    print("test_context_manager: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

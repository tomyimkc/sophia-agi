# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Context-manager goal-relevance retention (Prosoche efficiency payoff, thesis §5)."""
from __future__ import annotations

from agent.context_manager import ContextManager, Segment
from agent.prosoche import AttentionAnchor, anchor_segment, relevance_boost

ANCHOR = AttentionAnchor(
    goal="diagnose the slow checkout database query",
    in_scope_entities=("checkout", "database", "query"),
)


def _segs():
    return [
        anchor_segment(ANCHOR),
        Segment(kind="context", text="The checkout database query does a full table scan on orders. " * 6,
                provenance="on-goal"),
        Segment(kind="context", text="Unrelated: the office coffee machine schedule and lunch menu rota. " * 6,
                provenance="off-goal"),
    ]


def test_relevance_keeps_on_goal_over_off_goal_under_pressure():
    # Budget fits the anchor + exactly one of the two big segments. Relevance ranking
    # should admit the on-goal one and drop/compress the off-goal one.
    cm = ContextManager(120, relevance_fn=relevance_boost(ANCHOR))
    res = cm.pack(_segs())
    # On-goal context survives (kept whole or compressed); off-goal is dropped first.
    assert "on-goal" in res.kept or "on-goal" in res.compressed
    assert "off-goal" in res.dropped


def test_no_relevance_fn_is_unchanged_behaviour():
    # Without a relevance_fn the packer is byte-identical to before (priority order).
    segs = _segs()
    a = ContextManager(120).pack(segs)
    b = ContextManager(120, relevance_fn=None).pack(segs)
    assert a.text == b.text and a.cache_key == b.cache_key


def test_anchor_prefix_is_cache_stable():
    cm = ContextManager(500, relevance_fn=relevance_boost(ANCHOR))
    r1 = cm.pack(_segs())
    r2 = cm.pack(_segs())
    assert r1.cache_key == r2.cache_key
    assert r1.stable_prefix_tokens > 0


def test_relevance_never_evicts_pinned_anchor():
    cm = ContextManager(30, relevance_fn=relevance_boost(ANCHOR))
    res = cm.pack(_segs())  # tiny budget — only the pinned anchor survives
    assert any("anchor#" in k for k in res.kept)


def test_bad_relevance_fn_does_not_break_packing():
    def boom(_seg):
        raise RuntimeError("scorer exploded")

    cm = ContextManager(120, relevance_fn=boom)
    res = cm.pack(_segs())  # must still produce a pack, not raise
    assert res.tokens >= 0

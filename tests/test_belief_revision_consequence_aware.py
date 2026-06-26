#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.belief_revision_policy.resolve_conflicts_consequence_aware —
the LIVE consumer of reasoning.consequence.run_revise_loop.

These verify that routing real contradiction decisions through the ko-guarded loop
yields the load-bearing behavior: a bounded retraction cascade is accepted
(``allow``, no ko); a too-severe cascade surfaces the hesitation ko and escalates
(NEVER abstains); the single-pass ``resolve_conflicts`` ledger is preserved
verbatim alongside the consequence verdict; and the call is non-destructive.

Offline, deterministic (no model, no network). House pattern: sys.path bootstrap +
def main() + inspect runner; pytest also works.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.belief_revision_policy import (  # noqa: E402
    resolve_conflicts,
    resolve_conflicts_consequence_aware,
)
from okf.page import Page  # noqa: E402


def _page(pid: str, **meta) -> Page:
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept", **meta})


def _severe_pages() -> "list[Page]":
    # 'strong' (consensus) beats 'weak' (legendary) -> weak is the loser. weak
    # grounds a chain, so retracting it orphans 3/4 of the graph (severity 0.75).
    return [
        _page("strong", authorConfidence="consensus", contradicts=["weak"]),
        _page("weak", authorConfidence="legendary"),
        _page("d1", derivesFrom=["weak"]),
        _page("d2", derivesFrom=["d1"]),
    ]


def _bounded_pages() -> "list[Page]":
    # Same conflict, but 'weak' is a leaf inside a larger graph: retracting it
    # abstains only itself (1/12 < the 0.15 default threshold) -> bounded.
    base = [
        _page("strong", authorConfidence="consensus", contradicts=["weak"]),
        _page("weak", authorConfidence="legendary"),
    ]
    return base + [_page(f"n{i}") for i in range(10)]


def test_severe_cascade_kos_and_escalates_never_abstains() -> None:
    r = resolve_conflicts_consequence_aware(_severe_pages())
    assert r["consequenceVerdict"] == "escalate"
    assert r["consequenceVerdict"] != "abstain"  # the load-bearing invariant
    assert r["loop"]["ko"] is not None and r["loop"]["ko"]["ko"] is True
    assert r["loop"]["finalVerdict"] == "escalate"
    # the round-0 abstain set recurred (that is what makes it a ko)
    assert r["loop"]["roundsExecuted"] == 3
    # base ledger is still present: weak (+cascade) is the retracted side
    assert "weak" in r["retracted"]


def test_bounded_cascade_allows_without_ko() -> None:
    r = resolve_conflicts_consequence_aware(_bounded_pages())
    assert r["consequenceVerdict"] == "allow"
    assert r["loop"]["ko"] is None
    assert r["loop"]["roundsExecuted"] == 1
    assert "weak" in r["retracted"]


def test_no_conflict_allows_in_one_round() -> None:
    r = resolve_conflicts_consequence_aware([_page("a"), _page("b")])
    assert r["consequenceVerdict"] == "allow"
    assert r["loop"]["ko"] is None
    assert r["conflictCount"] == 0
    assert r["retracted"] == []


def test_base_resolve_conflicts_ledger_is_preserved() -> None:
    # The consequence-aware report must carry the single-pass ledger verbatim;
    # it augments resolve_conflicts, it does not change its decisions.
    pages = _severe_pages()
    base = resolve_conflicts(pages)
    aware = resolve_conflicts_consequence_aware(pages)
    for key in ("conflictCount", "conflicts", "kept", "retracted", "abstained"):
        assert aware[key] == base[key], f"ledger field {key} diverged from resolve_conflicts"
    # but it adds the consequence surface and re-tags the schema
    assert aware["schema"] == "sophia.belief_revision_policy.consequence_aware.v1"
    assert "consequenceVerdict" in aware and "loop" in aware


def test_abstain_verdict_conflict_has_no_retraction_consequence() -> None:
    # Two comparable beliefs -> resolve_conflicts abstains on both; there is no
    # loser to retract, so the consequence loop has an empty round-0 move -> allow.
    pages = [
        _page("claim_a", authorConfidence="attributed", contradicts=["claim_b"]),
        _page("claim_b", authorConfidence="attributed"),
    ]
    r = resolve_conflicts_consequence_aware(pages)
    assert r["abstained"] == ["claim_a", "claim_b"]
    assert r["retracted"] == []
    assert r["consequenceVerdict"] == "allow"
    assert r["loop"]["ko"] is None


def test_candidate_only_and_no_overclaim() -> None:
    r = resolve_conflicts_consequence_aware(_severe_pages())
    assert r["candidateOnly"] is True
    assert r["level3Evidence"] is False
    assert r["loop"]["candidateOnly"] is True
    assert r["loop"]["level3Evidence"] is False


def test_call_is_non_destructive() -> None:
    pages = _severe_pages()
    before_ids = [p.id for p in pages]
    before_meta = [dict(p.meta) for p in pages]
    resolve_conflicts_consequence_aware(pages)
    assert [p.id for p in pages] == before_ids
    assert [dict(p.meta) for p in pages] == before_meta


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_belief_revision_consequence_aware: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

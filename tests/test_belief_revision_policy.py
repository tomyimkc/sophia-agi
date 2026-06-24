#!/usr/bin/env python3
"""Tests for agent.belief_revision_policy — revise-or-abstain conflict resolution.

Verifies the AGM-style importance order (axiom > user > sourced > inferred), that a
weaker sourced belief yields on confidence, that comparable beliefs cause abstention
(never a silent overwrite), that the cascade is recorded, and that the policy beats
the last-write-wins baseline a weight model imitates. Offline, deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.belief_revision_policy import last_write_wins, resolve_conflicts  # noqa: E402
from okf.page import Page  # noqa: E402


def _page(pid: str, **meta) -> Page:
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept", **meta})


def test_higher_tier_wins_over_lower() -> None:
    pages = [
        _page("axiom_fact", beliefTier="axiom", authorConfidence="legendary", contradicts=["inferred_fact"]),
        _page("inferred_fact", beliefTier="inferred", authorConfidence="consensus"),
    ]
    report = resolve_conflicts(pages)
    assert report["conflicts"][0]["verdict"] == "kept"
    assert report["conflicts"][0]["kept"] == "axiom_fact"      # tier beats raw confidence
    assert "inferred_fact" in report["retracted"]
    assert "axiom_fact" in report["kept"]


def test_sourced_tie_breaks_on_confidence() -> None:
    pages = [
        _page("strong", authorConfidence="consensus", contradicts=["weak"]),
        _page("weak", authorConfidence="legendary"),
    ]
    report = resolve_conflicts(pages)
    assert report["conflicts"][0]["kept"] == "strong"
    assert "weak" in report["retracted"]


def test_comparable_beliefs_abstain_never_overwrite() -> None:
    # Two equal-tier, equal-confidence claims contradict: assert NEITHER.
    pages = [
        _page("claim_a", authorConfidence="attributed", contradicts=["claim_b"]),
        _page("claim_b", authorConfidence="attributed"),
    ]
    report = resolve_conflicts(pages)
    assert report["conflicts"][0]["verdict"] == "abstain"
    assert report["abstained"] == ["claim_a", "claim_b"]
    assert report["retracted"] == []


def test_cascade_recorded_on_retraction() -> None:
    # weak grounds a downstream claim; retracting weak must un-ground it (cascade).
    pages = [
        _page("strong", authorConfidence="consensus", contradicts=["weak"]),
        _page("weak", authorConfidence="legendary"),
        _page("downstream", derivesFrom=["weak"], authorConfidence="attributed"),
    ]
    report = resolve_conflicts(pages)
    conflict = report["conflicts"][0]
    assert "downstream" in conflict["cascade"]
    assert "downstream" in report["retracted"]


def test_beats_last_write_wins_baseline() -> None:
    # The older fact is higher-importance; a weight model (LWW) would forget it,
    # the policy keeps it.
    pages = [
        _page("old_axiom", beliefTier="axiom", authorConfidence="attributed"),
        _page("new_inferred", beliefTier="inferred", authorConfidence="consensus", contradicts=["old_axiom"]),
    ]
    baseline = last_write_wins(pages)
    policy = resolve_conflicts(pages)
    assert "old_axiom" in baseline["overwritten"]      # LWW forgets the axiom
    assert "old_axiom" in policy["kept"]               # policy preserves it
    assert "new_inferred" in policy["retracted"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

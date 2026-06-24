#!/usr/bin/env python3
"""Tests for agent.unlearning — reversible, audited forgetting on the OKF graph.

Verifies that forgetting a source un-grounds exactly its transitive support cascade,
that the runtime gate's abstain set covers those claims (and a grounded belief_state
no longer contains them), and that restore returns the belief state bit-for-bit —
the reversible unlearning a weight model cannot do. Offline, deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.unlearning import Unlearner  # noqa: E402
from okf.page import Page  # noqa: E402


def _page(pid: str, **meta) -> Page:
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept", **meta})


def _pages() -> "list[Page]":
    # poisoned <- derived_a <- derived_b ; independent stands alone
    return [
        _page("poisoned", authorConfidence="attributed"),
        _page("derived_a", derivesFrom=["poisoned"], authorConfidence="attributed"),
        _page("derived_b", derivesFrom=["derived_a"], authorConfidence="attributed"),
        _page("independent", authorConfidence="consensus"),
    ]


def test_forget_ungrounds_exact_cascade() -> None:
    u = Unlearner(_pages())
    res = u.forget("poisoned", reason="source forged")
    assert res.found is True
    assert set(res.blast_radius["supportLost"]) == {"derived_a", "derived_b"}
    # After forgetting, none of the abstained ids remain assertable.
    state = u.belief_state()
    for fid in ("poisoned", "derived_a", "derived_b"):
        assert fid not in state
    assert "independent" in state          # untouched fact survives


def test_abstain_set_is_what_gate_refuses() -> None:
    u = Unlearner(_pages())
    res = u.forget("poisoned", reason="debunked")
    assert set(res.abstain) == {"poisoned", "derived_a", "derived_b"}
    assert res.audit["event"] == "forget"
    assert res.audit["reason"] == "debunked"


def test_restore_is_bit_for_bit_reversible() -> None:
    u = Unlearner(_pages())
    before = u.belief_state()
    u.forget("poisoned", reason="mistake")
    assert u.belief_state() != before        # forgetting changed the belief state
    out = u.restore("poisoned")
    assert out["restored"] is True
    assert u.belief_state() == before        # round-trip exact
    assert u.tombstoned == []


def test_forget_unknown_source_is_fail_closed() -> None:
    u = Unlearner(_pages())
    res = u.forget("does_not_exist", reason="n/a")
    assert res.found is False
    assert res.id is None
    assert u.tombstoned == []                 # nothing silently removed


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

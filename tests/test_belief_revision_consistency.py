# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf/belief_revision_consistency.py — no orphaned belief survives a retraction."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okf.graph import Graph  # noqa: E402
from okf.belief_revision_consistency import (  # noqa: E402
    check_no_orphans_after_retraction,
    orphaned_beliefs,
)


def _node(nid: str, derives=None) -> dict:
    meta: dict = {}
    if derives is not None:
        meta["derivesFrom"] = list(derives)
    return {"id": nid, "pageType": "claim", "meta": meta, "page": None}


def _graph(spec: "dict") -> Graph:
    """Build a synthetic belief graph from {id: derivesFrom-list-or-None}."""
    g = Graph()
    for nid, der in spec.items():
        g.nodes[nid] = _node(nid, der)
    return g


def _base_graph() -> Graph:
    # src (primary source)
    # indep (independent primary source)
    # sole_dep derives ONLY from src            -> must be orphaned when src is retracted
    # multi_dep derives from src AND indep       -> must SURVIVE (indep still supports it)
    # grandchild derives from sole_dep           -> transitively orphaned when src goes
    return _graph({
        "src": None,
        "indep": None,
        "sole_dep": ["src"],
        "multi_dep": ["src", "indep"],
        "grandchild": ["sole_dep"],
    })


def test_retract_sole_support_flags_dependent():
    g = _base_graph()
    result = check_no_orphans_after_retraction(g, ["src"])
    # sole_dep lost its ONLY support -> must be flagged as an orphan.
    assert "sole_dep" in result["orphans"], result
    # grandchild transitively loses support too.
    assert "grandchild" in result["orphans"], result
    # The invariant must still hold: every orphan is covered by the abstain-set.
    assert result["ok"] is True, result
    assert set(result["orphans"]).issubset(set(result["abstain"]))
    assert result["uncovered"] == []


def test_belief_with_independent_support_survives():
    g = _base_graph()
    result = check_no_orphans_after_retraction(g, ["src"])
    # multi_dep still derives from indep, so it is NOT orphaned and NOT in abstain.
    assert "multi_dep" not in result["orphans"], result
    assert "multi_dep" not in result["abstain"], result
    # indep itself is untouched.
    assert "indep" not in result["orphans"]


def test_transitive_cascade_is_fully_covered():
    g = _base_graph()
    result = check_no_orphans_after_retraction(g, ["src"])
    # cascade (what revise reported) must cover the transitive orphans.
    assert "sole_dep" in result["cascade"]
    assert "grandchild" in result["cascade"]
    assert result["ok"] is True


def test_orphaned_beliefs_ground_truth_matches():
    g = _base_graph()
    # Direct ground-truth orphan computation (independent of revise()).
    orphans = orphaned_beliefs(g, {"src"})
    assert orphans == ["grandchild", "sole_dep"], orphans


def test_not_found_target_is_fail_closed():
    g = _base_graph()
    result = check_no_orphans_after_retraction(g, ["does_not_exist"])
    assert result["notFound"] == ["does_not_exist"], result
    # No real retraction happened -> nothing retracted, no orphans, invariant holds.
    assert result["retracted"] == []
    assert result["orphans"] == []
    assert result["ok"] is True


def test_retracting_independent_source_only_orphans_its_dependents():
    g = _base_graph()
    result = check_no_orphans_after_retraction(g, ["indep"])
    # indep is a sole support for nobody (multi_dep still has src), so no orphans.
    assert result["orphans"] == [], result
    assert result["ok"] is True


def test_no_orphan_when_primary_only_graph():
    # Two primary self-grounded claims; retracting one orphans nothing.
    g = _graph({"a": None, "b": None})
    result = check_no_orphans_after_retraction(g, ["a"])
    assert result["orphans"] == []
    assert result["ok"] is True


def _run():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    _run()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.incremental_revision — the incremental belief-revision engine.

The headline property is PROVABLE EQUIVALENCE to the full-recompute oracle
(agent.belief_revision_policy.resolve_conflicts): streaming facts through the
IncrementalReviser one at a time yields the SAME kept / retracted / abstained
partition the oracle computes over all pages at once — asserted deterministically
for several N and under permutation of a commutative case.

The sub-linearity claim is STRUCTURAL, not temporal: maxAffectedNodes (nodes actually
re-examined per update) stays bounded as N grows, where full recompute would touch ~N
each step. We assert on the affected-node curve and on compare_to_full(...)["subLinear"];
wall-time (time.perf_counter) is only reported by the harness, never asserted.

Offline, deterministic, dependency-free (standard library + the in-repo okf/agent only).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.incremental_revision import (  # noqa: E402
    IncrementalReviser,
    SUBLINEAR_TOLERANCE,
    compare_to_full,
    incremental_sweep,
)
from agent.belief_revision_policy import resolve_conflicts  # noqa: E402
from agent.belief_revision_scaling import make_pages  # noqa: E402
from okf.page import Page  # noqa: E402


def _page(pid: str, **meta) -> Page:
    """A self-grounded sourced fact unless overridden."""
    base = {"id": pid, "pageType": "concept", "authorConfidence": "attributed"}
    base.update(meta)
    return Page(path=Path(f"{pid}.md"), meta=base)


def _stream(pages) -> IncrementalReviser:
    reviser = IncrementalReviser()
    for page in pages:
        reviser.add_fact(page)
    return reviser


def _assert_equivalent(pages) -> None:
    """The incremental belief partition MUST equal the full-recompute oracle's."""
    reviser = _stream(pages)
    full = resolve_conflicts(pages)
    state = reviser.belief_state()
    assert reviser.kept_ids() == set(full["kept"]), (
        f"kept mismatch: inc={sorted(reviser.kept_ids())} full={full['kept']}"
    )
    assert set(state["kept"]) == set(full["kept"])
    assert set(state["retracted"]) == set(full["retracted"]), (
        f"retracted mismatch: inc={state['retracted']} full={full['retracted']}"
    )
    assert set(state["abstained"]) == set(full["abstained"]), (
        f"abstained mismatch: inc={state['abstained']} full={full['abstained']}"
    )


# --- Equivalence on the deterministic make_pages stream --------------------------

def test_equivalence_make_pages_n30() -> None:
    _assert_equivalent(make_pages(30))


def test_equivalence_make_pages_n60() -> None:
    _assert_equivalent(make_pages(60))


def test_belief_state_dict_is_fail_closed_candidate() -> None:
    reviser = _stream(make_pages(30))
    state = reviser.belief_state()
    # Every emitted verdict/decision dict carries candidateOnly: True.
    assert state["candidateOnly"] is True
    rep = reviser.add_fact(_page("late_fact"))
    assert rep["candidateOnly"] is True
    # A lone new fact with no contradiction touches only itself (bounded, not N).
    assert rep["affectedNodes"] == 1
    assert rep["affectedIds"] == ["late_fact"]


# --- Targeted scenario: axiom retracts a sourced fact + its dependents -----------

def test_axiom_retracts_sourced_fact_and_dependents_bounded() -> None:
    # b derivesFrom a; c derivesFrom b. An axiom x contradicts a: a is retracted and
    # b, c cascade (lose their provenance ground). The affected work is small (local
    # conflict + provenance chain), NOT proportional to the whole graph.
    a = _page("a")
    b = _page("b", derivesFrom=["a"])
    c = _page("c", derivesFrom=["b"])
    # Pad the graph with many unrelated, self-grounded facts so "small" is meaningful.
    filler = [_page(f"u{i}") for i in range(50)]

    reviser = IncrementalReviser()
    for page in [a, b, c, *filler]:
        reviser.add_fact(page)

    x = _page("x", contradicts=["a"], beliefTier="axiom")
    report = reviser.add_fact(x)

    # Exactly a (the contradicted sourced fact) and its transitive derivesFrom
    # dependents are retracted by this single update.
    assert set(report["deltaRetracted"]) == {"a", "b", "c"}
    assert report["deltaRestored"] == []

    # Final state retracts exactly a, b, c; everything else (incl. all filler) kept.
    assert reviser._retracted == {"a", "b", "c"}  # noqa: SLF001 - white-box assertion
    assert all(f"u{i}" in reviser.kept_ids() for i in range(50))

    # The affected subgraph is bounded by the local neighbourhood, NOT N: the new
    # axiom, the contradicted fact, and its derivesFrom dependents — not 50+ fillers.
    assert report["affectedNodes"] <= 6, report["affectedNodes"]
    assert report["affectedNodes"] < 50

    # And it matches the full oracle over the same page set.
    all_pages = [a, b, c, *filler, x]
    full = resolve_conflicts(all_pages)
    assert reviser.kept_ids() == set(full["kept"])
    assert reviser._retracted == set(full["retracted"])  # noqa: SLF001


# --- Structural sub-linearity: maxAffectedNodes bounded across N -----------------

def test_incremental_sweep_max_affected_bounded() -> None:
    sweep = incremental_sweep([30, 60, 120])
    assert [row["n"] for row in sweep] == [30, 60, 120]
    by_n = {row["n"]: row for row in sweep}

    smallest = by_n[30]["maxAffectedNodes"]
    largest = by_n[120]["maxAffectedNodes"]
    # The defining sub-linearity evidence: per-update affected work does NOT grow with
    # N — the largest-N maximum is within a small constant factor of the smallest-N
    # maximum. (Full recompute would touch ~N each step.) NO assertion on seconds.
    assert largest <= max(smallest, 1) * SUBLINEAR_TOLERANCE, (smallest, largest)

    # Every reported row exposes the structural measures (and is fail-closed).
    for row in sweep:
        assert row["maxAffectedNodes"] >= 1
        assert row["meanAffectedNodes"] >= 0.0
        assert row["totalAffected"] >= row["n"]  # at least the node itself each add
        assert row["seconds"] >= 0.0  # reported, not asserted-upon for a bound
        assert row["candidateOnly"] is True


def test_compare_to_full_verdict_sublinear() -> None:
    result = compare_to_full([30, 60, 120])
    assert result["subLinear"] is True
    assert result["candidateOnly"] is True
    # Both curves are present and aligned by size.
    assert [r["n"] for r in result["incremental"]] == [30, 60, 120]
    assert [r["n"] for r in result["full"]] == [30, 60, 120]
    assert result["tolerance"] == SUBLINEAR_TOLERANCE


# --- Order independence for a commutative case -----------------------------------

def test_order_independence_commutative_cascade() -> None:
    # a <- b <- c provenance chain, axiom x contradicts a. The final belief state is
    # insertion-order-independent (the retraction + cascade is commutative here), and
    # equals the full oracle for every permutation.
    pages = [
        _page("a"),
        _page("b", derivesFrom=["a"]),
        _page("c", derivesFrom=["b"]),
        _page("x", contradicts=["a"], beliefTier="axiom"),
    ]
    canonical = resolve_conflicts(pages)
    for perm in itertools.permutations(pages):
        reviser = _stream(list(perm))
        assert reviser.kept_ids() == set(canonical["kept"]), [p.id for p in perm]
        assert reviser._retracted == set(canonical["retracted"]), [p.id for p in perm]  # noqa: SLF001
        assert reviser._abstained == set(canonical["abstained"]), [p.id for p in perm]  # noqa: SLF001


def test_order_independence_abstain_tie() -> None:
    # Two same-tier sourced facts that contradict, equal confidence -> abstain BOTH,
    # regardless of which is inserted first (fail-closed: never silent-pick a side).
    pages = [_page("p"), _page("q", contradicts=["p"])]
    canonical = resolve_conflicts(pages)
    assert set(canonical["abstained"]) == {"p", "q"}
    for perm in itertools.permutations(pages):
        reviser = _stream(list(perm))
        assert reviser._abstained == {"p", "q"}, [p.id for p in perm]  # noqa: SLF001
        assert reviser.kept_ids() == set(canonical["kept"])


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

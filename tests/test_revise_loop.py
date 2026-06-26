#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Ko-guarded revise-loop tests — the iterative consumer of ``okf.revise``.

These exercise ``reasoning.consequence.run_revise_loop`` end-to-end over real OKF
graphs: a clean monotonic schedule (no ko), the canonical oscillation (ko -> escalate,
NEVER abstain), the ko-window boundary, a false-positive guard, non-destructiveness,
and the fail-closed path on an unresolved target.

Dependency-free, offline, deterministic (no model, no network).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from okf.page import Page  # noqa: E402
from reasoning.consequence import run_revise_loop  # noqa: E402


def _graph():
    # Two independent chains so retracts in different chains produce DIFFERENT
    # abstain sets (needed for a genuinely monotonic schedule).
    #   primary -> mid -> leaf   (chain A)
    #   root2 -> leaf2           (chain B)
    pages = [
        Page(path=Path("p.md"), meta={"id": "primary", "pageType": "concept", "authorConfidence": "consensus"}),
        Page(path=Path("m.md"), meta={"id": "mid", "pageType": "concept", "derivesFrom": ["primary"]}),
        Page(path=Path("l.md"), meta={"id": "leaf", "pageType": "concept", "derivesFrom": ["mid"]}),
        Page(path=Path("r2.md"), meta={"id": "root2", "pageType": "concept", "authorConfidence": "consensus"}),
        Page(path=Path("l2.md"), meta={"id": "leaf2", "pageType": "concept", "derivesFrom": ["root2"]}),
    ]
    return okf.build_graph(pages)


def test_monotonic_schedule_does_not_ko() -> None:
    # A schedule whose abstain sets are STRICTLY growing (each round retracts a
    # fresh node in a different chain, so the abstain set never recurs) must not
    # fire a ko. round1: retract primary (abstains chain A). round2: also retract
    # root2 (abstains chain A + chain B = strictly larger). round3: retract leaf2's
    # chain is already in; instead retract nothing new — but to keep it monotonic
    # we just use the two distinct-chain rounds, which cannot recur.
    g = _graph()
    st = run_revise_loop(g, retraction_schedule=[["primary"], ["primary", "root2"]])
    assert st.ko is None, f"monotonic schedule fired a ko: {st.ko}"
    assert st.terminated is False
    assert st.roundsExecuted == 2
    # abstain sets are distinct and the second is a strict superset of the first
    assert st.rounds[0] != st.rounds[1]
    assert st.rounds[0] < st.rounds[1]
    # no escalate-from-ko: final verdict is the last round's verdict (escalate here
    # because retracting both roots orphans 4/5 of the graph, but NOT from a ko)
    assert st.finalVerdict == st.verdicts[-1]


def test_canonical_oscillation_fires_ko_and_escalates() -> None:
    # The GO-ko analogue: retract A -> abstain set S; reassert (retract nothing) ->
    # abstain set {}; retract A again -> abstain set S recurs -> KO.
    g = _graph()
    st = run_revise_loop(g, retraction_schedule=[["primary"], [], ["primary"]])
    assert st.ko is not None and st.ko.ko is True, "oscillation must fire a ko"
    assert st.terminated is True
    assert st.roundsExecuted == 3  # the ko is detected on round 3 (the recurrence)
    # the load-bearing invariant: a ko escalates, it NEVER abstains
    assert st.finalVerdict == "escalate"
    assert st.finalVerdict != "abstain"
    # rounds 1 and 3 have the same abstain set (that's what makes it a ko)
    assert st.rounds[0] == st.rounds[2]
    assert st.ko.cycle[1] == 2  # recurrence detected at round index 2


def test_loop_is_non_destructive() -> None:
    # revise is non-destructive; the loop must not mutate the input graph either.
    g = _graph()
    before = set(g.nodes)
    run_revise_loop(g, retraction_schedule=[["primary"], [], ["primary"], ["root2"]])
    assert set(g.nodes) == before, "run_revise_loop mutated the input graph"


def test_ko_window_boundary_honors_ko_max_rounds() -> None:
    # A belief state recurring at gap (max_rounds+1) must NOT ko; at gap
    # max_rounds it MUST ko. Same schedule, two windows:
    #   round0: retract primary -> S
    #   round1: retract root2 -> S' (distinct)
    #   round2: retract leaf2 -> S'' (distinct)
    #   round3: retract primary again -> S recurs at gap 3
    sched = [["primary"], ["root2"], ["leaf2"], ["primary"]]
    # gap 3 > 2 -> no ko
    g1 = _graph()
    st_no = run_revise_loop(g1, retraction_schedule=sched, ko_max_rounds=2)
    assert st_no.ko is None, f"gap 3 with max_rounds=2 should not ko: {st_no.ko}"
    assert st_no.terminated is False
    # gap 3 <= 4 -> ko
    g2 = _graph()
    st_yes = run_revise_loop(g2, retraction_schedule=sched, ko_max_rounds=4)
    assert st_yes.ko is not None and st_yes.ko.ko is True, "gap 3 with max_rounds=4 should ko"
    assert st_yes.finalVerdict == "escalate"


def test_long_schedule_with_no_recurrence_does_not_false_positive() -> None:
    # A schedule longer than the window, where every round retracts a DIFFERENT
    # isolated node, must not false-positive a ko. Build a graph of N independent
    # nodes and retract one fresh node per round.
    pages = [Page(path=Path(f"n{i}.md"), meta={"id": f"n{i}", "pageType": "concept", "authorConfidence": "consensus"}) for i in range(8)]
    g = okf.build_graph(pages)
    sched = [[f"n{i}"] for i in range(6)]  # 6 rounds, each a distinct isolated node
    st = run_revise_loop(g, retraction_schedule=sched, ko_max_rounds=2)
    assert st.ko is None, f"distinct-node schedule false-positived a ko: {st.ko}"
    assert st.roundsExecuted == 6
    # all abstain sets are singletons of distinct nodes -> no two equal
    assert len(set(st.rounds)) == 6


def test_unresolved_target_terminates_fail_closed() -> None:
    # A round that retracts a ghost target cannot be bounded -> fail-closed abstain,
    # terminate immediately (we cannot reason about the cascade of a non-existent node).
    g = _graph()
    st = run_revise_loop(g, retraction_schedule=[["primary"], ["ghost_target"]])
    assert st.terminated is True
    assert "abstain" in st.verdicts
    assert st.finalVerdict == "abstain"
    assert "ghost_target" in st.reason


def test_state_schema_and_boundary() -> None:
    g = _graph()
    st = run_revise_loop(g, retraction_schedule=[["primary"]])
    d = st.to_dict()
    assert d["schema"] == "sophia.consequence.revise_loop.v1"
    assert d["candidateOnly"] is True and d["level3Evidence"] is False
    assert "AGI proof" in d["boundary"] or "not AGI" in d["boundary"]


def test_policy_mode_matches_equivalent_schedule() -> None:
    # A live policy that simply replays a fixed schedule must produce a result
    # identical to passing that schedule directly — the two modes share one round
    # engine; only the target source differs.
    sched = [["primary"], [], ["primary"]]

    def replay(i: int, prev_abstain: "frozenset[str]") -> "list[str] | None":
        return sched[i] if i < len(sched) else None

    g1, g2 = _graph(), _graph()
    st_sched = run_revise_loop(g1, retraction_schedule=sched)
    st_pol = run_revise_loop(g2, policy=replay)
    assert st_pol.rounds == st_sched.rounds
    assert st_pol.verdicts == st_sched.verdicts
    assert st_pol.finalVerdict == st_sched.finalVerdict == "escalate"
    assert st_pol.roundsExecuted == st_sched.roundsExecuted == 3
    assert (st_pol.ko is not None) and (st_sched.ko is not None)
    assert st_pol.ko.cycle == st_sched.ko.cycle


def test_requires_exactly_one_of_schedule_or_policy() -> None:
    g = _graph()
    # Neither given -> error.
    try:
        run_revise_loop(g)
        raise AssertionError("expected ValueError when neither mode is given")
    except ValueError:
        pass
    # Both given -> error (ambiguous).
    try:
        run_revise_loop(g, retraction_schedule=[["primary"]], policy=lambda i, p: None)
        raise AssertionError("expected ValueError when both modes are given")
    except ValueError:
        pass


def test_policy_safety_cap_fails_closed() -> None:
    # A policy that never stops and never recurs (a fresh isolated node every round)
    # cannot be bounded -> the loop must terminate fail-closed at max_policy_rounds.
    pages = [Page(path=Path(f"n{i}.md"), meta={"id": f"n{i}", "pageType": "concept", "authorConfidence": "consensus"}) for i in range(8)]
    g = okf.build_graph(pages)

    def never_stops(i: int, prev_abstain: "frozenset[str]") -> "list[str] | None":
        return [f"n{i % 8}"]

    st = run_revise_loop(g, policy=never_stops, max_policy_rounds=5, ko_max_rounds=4)
    assert st.terminated is True
    assert st.ko is None
    assert st.finalVerdict == "abstain"
    assert "max_policy_rounds" in st.reason


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_revise_loop: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

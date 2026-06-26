#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.lifelong_accumulation — the closed-loop lifelong-accumulation benchmark.

Verifies the HONEST metric end-to-end over a long synthetic stream: graph_backed
cumulative-correct net-accumulates (monotone non-decreasing modulo the deliberate
retraction) and ends strictly above the frozen parametric baseline's flat t0 count;
catastrophic (unintended) forgetting is 0 while deliberate unlearning is counted
separately; every fact in the graph was committed via the GovernedRSI cage and the
seeded poisoned + forbidden proposals were REJECTED with 0 cage breaches; the
control-flow gap (LexicalController) is reported and >= 0 with oracle >= routed; and
the report is byte-identical across two runs (deterministic). Offline, dependency-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.lifelong_accumulation import (  # noqa: E402
    accumulates_cleanly,
    make_lifelong_stream,
    run_accumulation,
)


def _report():
    return run_accumulation(make_lifelong_stream())


def test_report_shape_and_candidate_flags() -> None:
    r = _report()
    assert r["schema"] == "sophia.lifelong_accumulation.v1"
    assert r["candidateOnly"] is True
    assert r["validated"] is False
    for key in ("netCapabilityCurve", "finalGraphCorrect", "finalBaselineCorrect",
                "accumulates", "unintendedForgetting", "deliberateUnlearning",
                "cage", "controlFlowGap", "learningPriorities", "gapWorklist",
                "retentionMatrix", "citations"):
        assert key in r, key
    # The live multi-judge grader is a seam, never called offline.
    assert r["llmJudge"] is None


def test_net_capability_monotone_and_beats_baseline() -> None:
    r = _report()
    curve = r["netCapabilityCurve"]
    graph = [row["graphCorrectCumulative"] for row in curve]
    base = [row["baselineCorrectCumulative"] for row in curve]
    # Net accumulation: end strictly above the start AND above the flat baseline.
    assert graph[-1] > graph[0]
    assert graph[-1] > base[-1]
    # Monotone non-decreasing MODULO the deliberate retraction: any single-step
    # drop is bounded by the deliberately-unlearned count.
    deliberate = r["deliberateUnlearning"]
    for prev, cur in zip(graph, graph[1:]):
        if cur < prev:
            assert prev - cur <= deliberate


def test_no_catastrophic_forgetting_but_deliberate_is_counted() -> None:
    r = _report()
    assert r["unintendedForgetting"] == 0
    assert r["deliberateUnlearning"] > 0


def test_parametric_baseline_is_flat_at_t0_while_graph_grows() -> None:
    r = _report()
    curve = r["netCapabilityCurve"]
    base = [row["baselineCorrectCumulative"] for row in curve]
    graph = [row["graphCorrectCumulative"] for row in curve]
    # The frozen weight model knows only its t0 facts: it CANNOT learn, so its
    # cumulative-correct never rises — it is pinned to (at most) the t0 count and
    # can only fall when a t0 fact it cannot unlearn is retracted (it fabricates).
    assert base[0] == graph[0]                       # both correct on the t0 facts
    for prev, cur in zip(base, base[1:]):
        assert cur <= prev                           # never grows (cannot learn)
    assert max(base) == base[0]                      # flat-or-falling at the t0 count
    # ...while the graph-backed system genuinely grows far past it.
    assert graph[-1] > graph[0]
    assert r["finalBaselineCorrect"] <= base[0]
    assert r["finalGraphCorrect"] > r["finalBaselineCorrect"]


def test_cage_in_the_loop_admits_genuine_rejects_poison_and_forbidden() -> None:
    r = _report()
    cage = r["cage"]
    assert cage["breaches"] == 0
    assert cage["committed"] > 0
    assert cage["rejected"] >= 2
    assert cage["poisonRejected"] >= 1
    assert cage["killed"] is False
    # The seeded poisoned + forbidden proposals were rejected and never committed.
    rejected = set(cage["rejectedIds"])
    committed = set(cage["committedIds"])
    poison = [rid for rid in rejected if rid.startswith("poisoned_")]
    forbidden = [rid for rid in rejected if rid.startswith("forbidden_")]
    assert poison, "a poisoned proposal must be seeded and rejected"
    assert forbidden, "a forbidden/parametric proposal must be seeded and rejected"
    for bad in poison + forbidden:
        assert bad not in committed
    # Every cited fact in the graph corresponds to a committed unit (cage-gated).
    cited = {c["fact"] for c in r["citations"]}
    assert cited and cited.issubset(committed)


def test_skills_accumulate_through_the_cage() -> None:
    r = _report()
    committed = set(r["cage"]["committedIds"])
    skills = [cid for cid in committed if cid.startswith("skill_")]
    assert len(skills) >= 2, "at least a couple of skills should accumulate via the cage"


def test_control_flow_gap_reported_nonnegative_and_oracle_ge_routed() -> None:
    r = _report()
    gap = r["controlFlowGap"]
    cf = r["controlFlow"]
    assert isinstance(gap, float)
    assert gap >= 0.0
    assert cf["substrateAccuracy"] >= cf["endToEndAccuracy"]
    assert round(cf["substrateAccuracy"] - cf["endToEndAccuracy"], 4) == gap


def test_competence_and_gap_worklist_present() -> None:
    r = _report()
    assert isinstance(r["learningPriorities"], list)
    assert r["learningPriorities"], "measured-weakness ranking must be present"
    for item in r["learningPriorities"]:
        assert {"domain", "competence", "deficit", "reasons"} <= set(item)
    assert r["gapWorklist"]["schema"] == "sophia.knowledge_gap_worklist.v1"
    assert "worklist" in r["gapWorklist"]


def test_accumulates_cleanly_true() -> None:
    r = _report()
    assert accumulates_cleanly(r) is True


def test_report_is_deterministic_byte_identical() -> None:
    a = json.dumps(run_accumulation(make_lifelong_stream()), sort_keys=True)
    b = json.dumps(run_accumulation(make_lifelong_stream()), sort_keys=True)
    assert a == b


def test_accumulates_cleanly_rejects_a_breached_report() -> None:
    # Guard: accumulates_cleanly is a real check, not a constant True.
    r = _report()
    bad = dict(r)
    bad["cage"] = dict(r["cage"])
    bad["cage"]["breaches"] = 1
    assert accumulates_cleanly(bad) is False
    bad2 = dict(r)
    bad2["unintendedForgetting"] = 1
    assert accumulates_cleanly(bad2) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

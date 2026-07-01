#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Swarm-Router, its solo-vs-swarm benchmark, and its RLVR reward
(all offline, deterministic — no network, no torch)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import harness as h  # noqa: E402
from agent import model as m  # noqa: E402
from agent import swarm_router as sr  # noqa: E402
from provenance_bench import swarm_benchmark as sb  # noqa: E402
from provenance_bench import swarm_rl as srl  # noqa: E402


def _mock_client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


# --- router -----------------------------------------------------------------
def test_router_offline_invariants() -> None:
    ok, detail = sr.offline_invariants()
    assert ok, detail["checks"]


def test_trivial_is_solo_hard_is_swarm() -> None:
    r = sr.SwarmRouter()
    assert r.decide("hi").mode == "solo"
    plan = r.decide(
        "Compare the disputed authorship of the Dao De Jing versus the Analects, citing primary sources"
    )
    assert plan.mode == "swarm"
    assert plan.n_agents >= 2


def test_quant_task_gets_machine_verifier() -> None:
    plan = sr.SwarmRouter().decide("Prove that the probability of at least one six in four rolls exceeds 0.5")
    assert any(a.team == "math_verify" for a in plan.assignments)


def test_attribution_task_gets_ontology_cop() -> None:
    plan = sr.SwarmRouter().decide("Which ideas wrongly attributed to Freud actually originated later?")
    assert any(a.team == "ontology" for a in plan.assignments)


def test_plan_validates_against_schema_required_keys() -> None:
    schema = json.loads((ROOT / "schema" / "swarm-plan-1.0.0.json").read_text())
    required = set(schema["required"])
    d = sr.SwarmRouter().decide("Compare Kant and Hume on causation, citing sources").to_dict()
    assert required <= set(d)
    assert d["reduce"] == "fail_closed_synthesis"
    # every assignment carries the contract keys + a catalogue team
    teams_enum = set(schema["properties"]["assignments"]["items"]["properties"]["team"]["enum"])
    for a in d["assignments"]:
        assert {"team", "k", "budgetUsd", "goal"} <= set(a)
        assert a["team"] in teams_enum


def test_least_privilege_scope_is_subset() -> None:
    plan = sr.SwarmRouter().decide("Calculate 17*23 and verify the result")
    for spec in plan.to_specs():
        # each spec maps back to a team whose scope it must not exceed
        team = next(t for t in sr.TEAMS.values() if spec.label.startswith(t.name))
        if team.allowed_tools is None:
            assert spec.allowed_tools is None
        else:
            assert spec.allowed_tools is not None
            assert set(spec.allowed_tools) <= set(team.allowed_tools)


def test_route_batch_utilisation_normalised() -> None:
    r = sr.SwarmRouter()
    batch = sr.route_batch(r, [
        "hi", "Compare Kant and Hume citing sources", "Calculate 12*13 and verify",
        "Which quote is misattributed to Einstein?",
    ])
    if batch["totalAgents"]:
        assert abs(sum(batch["teamFraction"].values()) - 1.0) < 1e-9
    assert 0.0 <= batch["soloRate"] <= 1.0


def test_run_swarm_isolated_and_failclosed_reduce() -> None:
    """The router's plan executes through the real delegation layer: children are
    isolated, the parent trace brackets them, and the reduce is fail-closed —
    n_ok==0 must yield the ABSTAIN synthesis, never an invented answer. (This task's
    'legendary authorship' content is correctly abstained on by the source gate, so it
    also exercises the fail-closed path.)"""
    from agent import subagent as sa

    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        plan, result = sr.run_swarm(
            "Compare the disputed authorship of the Dao De Jing versus the Analects, citing sources",
            client=_mock_client(), parent_id="t1",
        )
        assert plan.mode == "swarm"
        assert len(result.children) == plan.n_agents
        # children are isolated (distinct trace files)
        assert len({c.trace_path for c in result.children}) == len(result.children)
        # parent delegation trace brackets the children
        events = [json.loads(l) for l in Path(result.trace_path).read_text().splitlines() if l.strip()]
        types = [e["type"] for e in events]
        assert types[0] == "delegate_start" and types[-1] == "delegate_end"
        # fail-closed reduce invariant (holds regardless of how many children passed the gate)
        if result.n_ok == 0:
            assert result.synthesis == sa.ABSTAIN_NO_CHILDREN and not result.ok
        else:
            assert result.ok and result.synthesis.strip()


def test_run_swarm_solo_path_succeeds() -> None:
    """A trivial task routes solo and completes through an isolated backbone child."""
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        plan, result = sr.run_swarm("summarize the plan", client=_mock_client(), parent_id="t2")
        assert plan.mode == "solo"
        assert len(result.children) == 1
        assert result.ok is True and result.n_ok == 1


# --- benchmark --------------------------------------------------------------
def test_benchmark_offline_invariants() -> None:
    ok, detail = sb.offline_invariants()
    assert ok, detail["checks"]


def test_benchmark_no_false_win_when_swarm_useless() -> None:
    # A swarm solver identical to solo must NOT register a win (CI includes zero).
    tasks = [sb.Task(f"Compare claim {i} vs rival citing sources", "X", hard=True) for i in range(10)]
    rep = sb.run_benchmark(
        tasks,
        solve_solo=lambda t: "Y",
        solve_swarm=lambda t, plan: "Y",  # same as solo → no advantage
        verify=lambda a, g: a == g,
    )
    assert not rep.is_win
    assert rep.delta == 0.0


# --- RLVR reward ------------------------------------------------------------
def test_reward_offline_invariants() -> None:
    ok, detail = srl.offline_invariants()
    assert ok, detail["checks"]


def test_reward_punishes_gate_failure_below_solo() -> None:
    r = sr.SwarmRouter()
    solo = r.decide("hi")
    swarm = r.decide("Compare the disputed authorship of the Dao De Jing versus the Analects, citing sources")
    clean_solo = srl.swarm_reward(srl.SwarmOutcome(solo, verified_success=1.0))
    leaky_swarm = srl.swarm_reward(
        srl.SwarmOutcome(swarm, verified_success=1.0, n_agents_failed_gate=swarm.n_agents)
    )
    assert clean_solo > leaky_swarm


def test_reward_bounded() -> None:
    r = sr.SwarmRouter()
    plan = r.decide("Compare disputed claims citing sources and adjudicate counter-arguments")
    worst = srl.swarm_reward(
        srl.SwarmOutcome(plan, verified_success=0.0, n_agents_failed_gate=plan.n_agents, serial_depth=20),
        load_imbalance=len(sr.TEAMS),
    )
    assert srl.REWARD_FLOOR <= worst <= srl.REWARD_CEIL


# --- multi-turn trajectory reward (KL control + length normalisation) -------
def test_trajectory_offline_invariants() -> None:
    ok, detail = srl.trajectory_invariants()
    assert ok, detail["checks"]


def test_trajectory_kl_control_penalises_drift() -> None:
    r = sr.SwarmRouter()
    step = r.decide("Compare the disputed authorship of the Dao De Jing citing sources")
    turns = [srl.SwarmOutcome(step, verified_success=1.0) for _ in range(3)]
    calm = srl.trajectory_reward(srl.TrajectoryOutcome(turns, 1.0, kl_per_turn=(0.0, 0.0, 0.0)))
    drift = srl.trajectory_reward(srl.TrajectoryOutcome(turns, 1.0, kl_per_turn=(3.0, 3.0, 3.0)))
    assert calm > drift


def test_trajectory_length_normalisation_reduces_cost() -> None:
    r = sr.SwarmRouter()
    step = r.decide("Compare the disputed authorship of the Dao De Jing citing sources")
    turns = [srl.SwarmOutcome(step, verified_success=1.0) for _ in range(4)]
    normed = srl.trajectory_reward(srl.TrajectoryOutcome(turns, 1.0), length_normalize=True)
    raw = srl.trajectory_reward(srl.TrajectoryOutcome(turns, 1.0), length_normalize=False)
    assert normed > raw


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [ok] {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {fn.__name__}")
            traceback.print_exc()
    print(f"{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)

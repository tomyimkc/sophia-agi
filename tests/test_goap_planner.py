#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the GOAP planner (preconditions/effects, bounded A*, replan) and its
lowering into the long-horizon ledger. Deterministic, offline, stdlib-only."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import long_horizon as lh  # noqa: E402
from agent.goap_planner import (  # noqa: E402
    Action,
    _demo_actions,
    offline_invariants,
    plan,
    plan_to_subtasks,
    replan,
)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail


def test_cheapest_plan_wins() -> None:
    """A* picks the cheaper route when two paths reach the goal."""
    actions = [
        Action("slow", add=frozenset({"done"}), cost=3.0),
        Action("prep", add=frozenset({"ready"})),
        Action("fast", preconditions=frozenset({"ready"}), add=frozenset({"done"})),
    ]
    p = plan(frozenset(), frozenset({"done"}), actions)
    assert p is not None and [a.name for a in p.actions] == ["prep", "fast"]
    assert p.cost == 2.0


def test_delete_effects_are_modeled() -> None:
    """An action that consumes a resource forces re-establishment."""
    actions = [
        Action("make_token", add=frozenset({"token"})),
        Action("spend_a", preconditions=frozenset({"token"}),
               add=frozenset({"a"}), delete=frozenset({"token"})),
        Action("spend_b", preconditions=frozenset({"token"}),
               add=frozenset({"b"}), delete=frozenset({"token"})),
    ]
    p = plan(frozenset(), frozenset({"a", "b"}), actions)
    assert p is not None
    assert [a.name for a in p.actions].count("make_token") == 2


def test_replan_event_is_ledger_shaped() -> None:
    ev = replan(frozenset({"sources:collected"}),
                frozenset({"gate:claims:passed", "report:published"}),
                _demo_actions(), failed_action="draft", abandoned=["draft"])
    assert ev["schema"] == "sophia.goap_replan.v1"
    assert ev["failedAction"] == "draft"
    assert ev["abandonedPlan"] == ["draft"]
    assert ev["currentState"] == ["sources:collected"]
    # goal reachable only if the resource claim is held → this one is held
    assert ev["replanned"] is False and ev["verdict"] == "held"


def test_plan_lowers_into_runnable_ledger() -> None:
    """plan_to_subtasks output is accepted verbatim by build_ledger and runs
    end-to-end on the mock client, in plan order."""
    from agent import model as m

    p = plan(frozenset({"resource:ci:free"}),
             frozenset({"report:published", "gate:claims:passed"}), _demo_actions())
    assert p is not None
    with tempfile.TemporaryDirectory() as tmp:
        ledger = lh.build_ledger("publish the report", plan_to_subtasks(p),
                                 ledger_id="goap-e2e", ledgers_dir=Path(tmp))
        result = lh.run_long_horizon(
            ledger, client=m.ModelClient(m.resolve_config("mock")),
            recovery=lh.RecoveryMemory(path=Path(tmp) / "r.jsonl"))
        assert result.ok
        assert result.completed == [s["id"] for s in plan_to_subtasks(p)]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the Data Analysis Agent (Phase 5) + its swarm-router seam."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.data_analyst import DataAnalyst, SCORE_FLOOR  # noqa: E402
from agent import swarm_router as sr  # noqa: E402


def test_assess_is_ok_and_honest() -> None:
    a = DataAnalyst().assess()
    assert a["status"] == "ok"
    assert a["canClaimAGI"] is False
    assert a["proposeOnly"] is True
    assert 0.0 <= a["dhi"] <= 1.0
    assert set(a["dimensions"]) == set(a["weights"])


def test_assess_is_deterministic() -> None:
    assert DataAnalyst().assess() == DataAnalyst().assess()


def test_curation_plan_only_flags_weak_dimensions_sorted_by_priority() -> None:
    plan = DataAnalyst().curation_plan()
    assert plan["status"] == "ok"
    prios = [a["priority"] for a in plan["actions"]]
    assert prios == sorted(prios, reverse=True), "actions must be priority-desc"
    for act in plan["actions"]:
        if act["dimension"] in DataAnalyst().assess()["dimensions"]:
            assert act["score"] < SCORE_FLOOR


def test_plan_surfaces_entity_contamination_action() -> None:
    plan = DataAnalyst().curation_plan()
    dims = {a["dimension"] for a in plan["actions"]}
    # the repo's known SEIB contamination must surface as a carve action
    assert "entityDisjointSplit" in dims


def test_adopted_split_demotes_carve_action_and_points_to_third_party() -> None:
    # Once the entity-disjoint split is sealed + adopted, the plan must stop proposing
    # another carve and instead surface the real residual (a third-party pack). The action
    # stays present (the broad eval is still contaminated) but is no longer top priority.
    from agent.data_analyst import _adopted_entity_split
    if _adopted_entity_split(ROOT) is None:
        return  # split not adopted in this checkout — nothing to assert
    plan = DataAnalyst().curation_plan()
    eds = next(a for a in plan["actions"] if a["dimension"] == "entityDisjointSplit")
    assert "third-party" in eds["recommendation"].lower()
    assert "carve_entity_disjoint_split" not in eds["tool"]
    assert plan["topPriority"] != "entityDisjointSplit", "a carved+adopted split is not top priority"


def test_fail_closed_when_manifest_missing(monkeypatch) -> None:
    from tools import data_health_report as dhr
    monkeypatch.setattr(dhr, "MANIFEST", ROOT / "does-not-exist" / "manifest.json")
    a = DataAnalyst().assess()
    assert a["status"] == "refused"
    # a refused assessment propagates through the plan (never a fabricated plan)
    assert DataAnalyst().curation_plan(a)["status"] == "refused"


def test_report_combines_assessment_and_plan() -> None:
    rep = DataAnalyst().report()
    assert rep["status"] == "ok"
    assert rep["assessment"]["status"] == "ok"
    assert rep["plan"]["status"] == "ok"


# --- swarm-router seam ------------------------------------------------------
def test_data_team_registered_and_least_privilege() -> None:
    assert "data" in sr.TEAMS
    team = sr.TEAMS["data"]
    assert team.allowed_tools is not None and set(team.allowed_tools) <= {"python", "retrieve"}


def test_data_task_routes_to_data_team() -> None:
    plan = sr.SwarmRouter().decide("audit our training corpus for decontamination and data quality")
    assert plan.mode == "swarm"
    assert any(a.team == "data" for a in plan.assignments)


def test_short_data_task_still_spawns_data_team() -> None:
    # verifiability of the data process doesn't scale with query length
    plan = sr.SwarmRouter().decide("decontaminate the dataset")
    assert any(a.team == "data" for a in plan.assignments)


def test_data_team_in_plan_schema_enum() -> None:
    import json
    schema = json.loads((ROOT / "schema" / "swarm-plan-1.0.0.json").read_text())
    enum = schema["properties"]["assignments"]["items"]["properties"]["team"]["enum"]
    assert "data" in enum

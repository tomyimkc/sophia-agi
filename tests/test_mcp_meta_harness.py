#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the meta-harness MCP tools (capabilities index, gated memory tools,
route preview, trajectory capture, resource claims) — impl layer, offline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_mcp import tools_impl as impl  # noqa: E402
from sophia_mcp.audit import TOOL_RISK  # noqa: E402


def test_capabilities_indexes_every_served_tool() -> None:
    caps = impl.capabilities()
    assert caps["schema"] == "sophia.capabilities.v1"
    indexed = sum(len(v) for v in caps["families"].values())
    assert indexed == caps["nTools"] > 60
    # the new meta-harness tools index themselves
    all_tools = {t["tool"] for fam in caps["families"].values() for t in fam}
    assert {"sophia_capabilities", "sophia_memory_search", "sophia_memory_store",
            "sophia_route_task", "sophia_trajectory_record",
            "sophia_claim_resource"} <= all_tools


def test_write_shaped_meta_tools_are_risk_covered() -> None:
    for tool in ("sophia_memory_store", "sophia_trajectory_record",
                 "sophia_claim_resource"):
        assert TOOL_RISK.get(tool) == "medium", tool


def test_memory_store_requires_approval_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOPHIA_MCP_APPROVE_WRITES", raising=False)
    out = impl.memory_store({"task": "t", "outcomeSummary": "s",
                             "verdict": "accepted", "verifiedBy": ["gate"]})
    assert out.get("denied") is True and "SOPHIA_MCP_APPROVE_WRITES" in out["error"]


def test_memory_store_gate_quarantines_unverified(monkeypatch: pytest.MonkeyPatch,
                                                  tmp_path: Path) -> None:
    monkeypatch.setenv("SOPHIA_MCP_APPROVE_WRITES", "1")
    # point the bank at a temp store so the test never touches real memory
    from agent import experience_memory as em

    monkeypatch.setattr(em, "BANK_PATH", tmp_path / "bank.jsonl")
    monkeypatch.setattr(em, "QUARANTINE_PATH", tmp_path / "q.jsonl")
    held = impl.memory_store({"task": "t", "outcomeSummary": "s",
                              "verdict": "accepted"})  # no verifiedBy
    assert held["verdict"] == "held"
    stored = impl.memory_store({"task": "check the Analects attribution",
                                "outcomeSummary": "verified against corpus",
                                "verdict": "accepted", "verifiedBy": ["gate"]})
    assert stored["verdict"] == "accepted"
    hits = impl.memory_search("Analects attribution")
    assert hits["matches"] and "advisory" in hits["note"]


def test_route_task_preview_is_contract_shaped() -> None:
    plan = impl.route_task_preview("Compare Kant and Hume on causation in detail")
    assert plan["schema"] == "sophia.swarm_plan.v1"
    assert plan["mode"] in ("solo", "swarm")
    # read-only preview carries the honest cost estimate
    assert "estCostSteps" in plan


def test_trajectory_record_schema_checked(monkeypatch: pytest.MonkeyPatch,
                                          tmp_path: Path) -> None:
    monkeypatch.setenv("SOPHIA_MCP_APPROVE_WRITES", "1")
    monkeypatch.setattr(impl, "SESSION_TRACES", tmp_path / "events.jsonl")
    bad = impl.trajectory_record({"no_kind": True})
    assert bad["ok"] is False
    good = impl.trajectory_record({"kind": "session_note", "note": "unit"})
    assert good["ok"] is True
    row = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))
    assert row["kind"] == "session_note" and "ts" in row


def test_claim_resource_actions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SOPHIA_MCP_APPROVE_WRITES", "1")
    from agent import resource_claims as rc

    monkeypatch.setattr(rc, "CLAIMS_PATH", tmp_path / "claims.json")
    assert impl.claim_resource("claim", "spark-gpu", "sess-1", 60)["ok"]
    st = impl.claim_resource("status")
    assert st["ok"] and "spark-gpu" in st["claims"]
    assert impl.claim_resource("heartbeat", "spark-gpu", "sess-1")["ok"]
    assert impl.claim_resource("release", "spark-gpu", "sess-1")["ok"]
    assert not impl.claim_resource("frobnicate")["ok"]

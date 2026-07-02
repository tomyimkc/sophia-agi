#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the agent-readiness audit (deterministic scorecard over the harness)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.agent_readiness_audit import _CONTRACT_GOVERNED, audit  # noqa: E402


def test_audit_runs_clean_on_this_checkout() -> None:
    """The committed repo must audit with zero FAILs (WARNs allowed — e.g. a locked
    git-crypt clone). A FAIL here means the harness itself regressed."""
    checks = audit()
    fails = [c for c in checks if c["status"] == "FAIL"]
    assert not fails, fails


def test_audit_covers_the_named_surfaces() -> None:
    names = {c["name"] for c in audit()}
    assert {"mcp.config", "mcp.risk_coverage", "gateway.flags", "hooks.scripts",
            "skills.readable", "claims.store", "rag.index", "memory.writable"} <= names


def test_exemption_list_is_closed() -> None:
    """The contract-governed exemption list must not silently grow — any new
    write-shaped tool needs @audited(risk=...), not an exemption."""
    assert _CONTRACT_GOVERNED == {"sophia_enqueue_task", "sophia_record_claim",
                                  "sophia_retract", "sophia_revise"}


def test_gateway_flags_reported_when_enabled() -> None:
    checks = audit(env={"SOPHIA_MCP_GATEWAY": "1"})
    flag = next(c for c in checks if c["name"] == "gateway.flags")
    assert "SOPHIA_MCP_GATEWAY" in flag["detail"]

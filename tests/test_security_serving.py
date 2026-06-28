# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for activating the hardened security profile in the MCP server
entrypoints behind SOPHIA_HARDENED=1. Skipped where MCP deps aren't installed
(the import of the server modules requires fastmcp)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_gateway_server_plain_by_default(monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.delenv("SOPHIA_HARDENED", raising=False)
    import gateway.server as gs
    importlib.reload(gs)
    assert gs._GW.output_guard is False
    assert gs._GW.audit_log is None


def test_gateway_server_hardened_when_flagged(monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("SOPHIA_HARDENED", "1")
    monkeypatch.setenv("SOPHIA_CANARY_SEED", "test-seed")
    import gateway.server as gs
    importlib.reload(gs)
    assert gs._GW.output_guard is True          # egress leak guard active
    assert gs._GW.audit_log is not None         # tamper-evident trail active
    assert gs._GW.call_budget is not None       # anti-DoS budget active
    # echo detection wired to the advertised instructions
    assert gs._GW.system_prompt == gs._INSTRUCTIONS
    importlib.reload(gs)  # leave module in a clean state for other tests


def test_build_gateway_without_seed_falls_back(monkeypatch):
    pytest.importorskip("mcp")
    monkeypatch.setenv("SOPHIA_HARDENED", "1")
    monkeypatch.delenv("SOPHIA_CANARY_SEED", raising=False)
    import gateway.server as gs
    importlib.reload(gs)
    gw = gs._build_gateway()
    assert gw.output_guard is True
    assert gw.canaries is None                  # no seed → shape-based detection

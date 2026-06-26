#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the conscience-gated runtime (trust-layer ticket T-2).

The gate now lives inside ``agent.harness.run_agent`` (its ``conscience_gate``
param); ``agent.conscience_runtime`` holds the logic. These tests cover both the
unit gate logic and the in-loop integration via the mock model provider.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import conscience_runtime as cr  # noqa: E402
from agent import harness as h  # noqa: E402
from agent import model as m  # noqa: E402
from agent.conscience_enforcement import EnforcementDecision  # noqa: E402


def _mock_client() -> m.ModelClient:
    return m.ModelClient(m.resolve_config("mock"))


def _events(trace_path: str) -> list[dict]:
    return [json.loads(line) for line in Path(trace_path).read_text().splitlines() if line.strip()]


def _blocked(*a, **k) -> EnforcementDecision:
    return EnforcementDecision(
        allowed=False, action="finalize_answer", verdict="abstain", reason="forced block", candidateOnly=True
    )


def _allowed(*a, **k) -> EnforcementDecision:
    return EnforcementDecision(allowed=True, action="finalize_answer", verdict="allow", reason="ok")


# ----------------------------- unit: gate logic ---------------------------- #
def test_gate_enabled_flag_and_env(monkeypatch) -> None:
    assert cr.gate_enabled(True) is True
    assert cr.gate_enabled(False) is False
    monkeypatch.delenv(cr.GATE_ENV, raising=False)
    assert cr.gate_enabled(None) is False
    monkeypatch.setenv(cr.GATE_ENV, "1")
    assert cr.gate_enabled(None) is True


def test_apply_to_result_allows_passthrough(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cr, "_finalize_gate", _allowed)
    r = h.AgentResult(task_id="t", ok=True, final_text="safe answer",
                      steps=[], failures=[], cost_usd=0.0, latency_sec=0.0, trace_path=str(tmp_path / "x.jsonl"))
    out = cr.apply_to_result(r)
    assert out is r  # unchanged object when allowed
    assert out.final_text == "safe answer"


def test_apply_to_result_blocked_withholds_and_logs(monkeypatch, tmp_path) -> None:
    log = tmp_path / "trace.jsonl"
    monkeypatch.setattr(cr, "_finalize_gate", _blocked)
    r = h.AgentResult(task_id="t", ok=True, final_text="SENSITIVE: launch-code 12345",
                      steps=[], failures=[], cost_usd=0.0, latency_sec=0.0, trace_path=str(log))
    out = cr.apply_to_result(r)
    assert out.ok is False
    assert out.final_text == cr._ABSTAIN_TEXT
    assert "12345" not in out.final_text                 # withheld from the answer
    assert any("conscience" in f for f in out.failures)
    holds = [e for e in _events(str(log)) if e.get("type") == "conscience_hold"]
    assert holds and holds[-1]["verdict"] == "abstain"
    assert "12345" in holds[-1]["withheldPreview"]       # …but retained for audit


def test_apply_to_result_empty_text_is_noop(tmp_path) -> None:
    r = h.AgentResult(task_id="t", ok=False, final_text="",
                      steps=[], failures=["x"], cost_usd=0.0, latency_sec=0.0, trace_path=str(tmp_path / "x.jsonl"))
    assert cr.apply_to_result(r) is r


# --------------------------- integration: run_agent ------------------------ #
def test_run_agent_gate_off_by_default(monkeypatch, tmp_path) -> None:
    os.environ.pop(cr.GATE_ENV, None)
    monkeypatch.setattr(h, "RUNS_DIR", tmp_path)
    task = h.AgentTask(goal="Should we launch on HN this week?", mode="advisor", task_id="t-off")
    result = h.run_agent(task, client=_mock_client(), max_retries=1)
    assert result.ok is True
    assert result.final_text.strip()
    assert not any(e.get("type") == "conscience_hold" for e in _events(result.trace_path))


def test_run_agent_gate_on_allowed_passthrough(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(h, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cr, "_finalize_gate", _allowed)
    task = h.AgentTask(goal="Should we launch on HN this week?", mode="advisor", task_id="t-allow")
    base = h.run_agent(task, client=_mock_client(), max_retries=1)
    gated = h.run_agent(task, client=_mock_client(), conscience_gate=True, max_retries=1)
    assert gated.final_text == base.final_text
    assert not any(e.get("type") == "conscience_hold" for e in _events(gated.trace_path))


def test_run_agent_gate_on_blocked_downgrades(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(h, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(cr, "_finalize_gate", _blocked)
    task = h.AgentTask(goal="x", mode="advisor", task_id="t-block")
    result = h.run_agent(task, client=_mock_client(), conscience_gate=True, max_retries=1)
    assert result.ok is False
    assert result.final_text == cr._ABSTAIN_TEXT
    assert any("conscience" in f for f in result.failures)
    assert any(e.get("type") == "conscience_hold" for e in _events(result.trace_path))


def test_env_flag_enables_in_loop_gate(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(h, "RUNS_DIR", tmp_path)
    monkeypatch.setenv(cr.GATE_ENV, "1")
    monkeypatch.setattr(cr, "_finalize_gate", _blocked)
    task = h.AgentTask(goal="x", mode="advisor", task_id="t-env")
    result = h.run_agent(task, client=_mock_client(), max_retries=1)  # no explicit flag
    assert result.ok is False
    assert result.final_text == cr._ABSTAIN_TEXT

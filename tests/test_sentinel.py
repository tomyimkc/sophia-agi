# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.sentinel (offline supervision) + sophia_mcp.approval (gate)
+ skills.sentinel_watch — SophiaArk Phase 2.

Invariants:
  1. failure classification mirrors the harness vocabulary, deterministically.
  2. drift detection is pure, bounded, and flags an abstention collapse.
  3. the approval gate is default-OFF (byte-identical served surface) and, when
     on, HOLDS before dispatch (no side effect) and never stores raw args.
  4. the sentinel_watch skill is fail-closed.
"""
from __future__ import annotations

import json
import os
from collections import Counter

from agent import sentinel
from agent.harness import FAILURE_CLASSES
from skills import run_skill


# --------------------------------------------------------------------------- #
# 1. Failure classification + aggregation
# --------------------------------------------------------------------------- #

def test_classify_failure_mirrors_harness_vocabulary():
    assert sentinel.classify_failure(
        {"type": "step_output", "passed": False, "failureClass": "gate_violation"}
    ) == "gate_violation"
    assert sentinel.classify_failure(
        {"type": "step_output", "passed": True, "failureClass": None}
    ) is None
    assert sentinel.classify_failure(
        {"type": "critic", "gatePassed": False, "verifierPassed": True}
    ) == "gate_violation"
    # an out-of-vocabulary class is normalised to "unknown", never invented
    assert sentinel.classify_failure(
        {"type": "step_output", "passed": False, "failureClass": "weird"}
    ) == "unknown"
    assert "gate_violation" in FAILURE_CLASSES


def test_scan_events_counts_steps_failures_and_savings():
    events = [
        {"type": "step_output", "passed": True},
        {"type": "step_output", "passed": False, "failureClass": "verifier_fail"},
        {"type": "critic", "gatePassed": False, "verifierPassed": False},
        {"type": "arkdistill", "savedTokens": 120},
        {"type": "arkdistill", "savedTokens": 30},
    ]
    rep = sentinel.scan_events(events)
    assert rep.steps == 2
    assert rep.failures["verifier_fail"] == 1
    assert rep.saved_tokens == 150 and rep.arkdistill_events == 2
    assert 0.0 <= rep.failure_rate <= 1.0
    # deterministic to_dict
    assert sentinel.scan_events(events).to_dict() == rep.to_dict()


# --------------------------------------------------------------------------- #
# 2. Drift detection
# --------------------------------------------------------------------------- #

def test_detect_drift_flags_abstention_collapse():
    baseline = Counter({"accepted": 50, "held": 50})   # 50% abstention
    recent = Counter({"accepted": 95, "held": 5})       # 5% abstention — collapse
    d = sentinel.detect_drift(recent, baseline, threshold=0.15)
    assert d["drift"] is True
    assert d["direction"] == "abstention_drop"
    assert 0.0 <= d["l1"] <= 1.0 and 0.0 <= d["abstentionDelta"] <= 1.0


def test_detect_drift_quiet_when_stable():
    baseline = Counter({"accepted": 50, "held": 50})
    recent = Counter({"accepted": 52, "held": 48})
    assert sentinel.detect_drift(recent, baseline, threshold=0.15)["drift"] is False


def test_detect_drift_is_deterministic_and_handles_empty():
    assert sentinel.detect_drift({}, {})["drift"] is False
    a = sentinel.detect_drift(Counter({"held": 3, "accepted": 1}), Counter({"held": 1, "accepted": 3}))
    b = sentinel.detect_drift(Counter({"held": 3, "accepted": 1}), Counter({"held": 1, "accepted": 3}))
    assert a == b


def test_scan_runs_reads_jsonl_and_is_fail_open(tmp_path):
    log = tmp_path / "run-1.jsonl"
    log.write_text(
        "\n".join([
            json.dumps({"type": "step_output", "passed": False, "failureClass": "tool_error"}),
            "{ this is not json",  # bad line must be skipped, not fatal
            json.dumps({"type": "arkdistill", "savedTokens": 77}),
        ]),
        encoding="utf-8",
    )
    rep = sentinel.scan_runs(tmp_path)
    assert rep.runs == 1 and rep.failures["tool_error"] == 1 and rep.saved_tokens == 77
    # missing dir => empty report, never raises
    assert sentinel.scan_runs(tmp_path / "nope").runs == 0


# --------------------------------------------------------------------------- #
# 3. Approval gate (default off; holds before dispatch; secret-safe)
# --------------------------------------------------------------------------- #

def test_approval_gate_is_default_off(monkeypatch):
    monkeypatch.delenv("SOPHIA_MCP_APPROVAL", raising=False)
    from sophia_mcp import approval
    assert approval.approval_enabled() is False
    assert approval.requires_approval("sophia_wiki_upsert") is False


def test_approval_gate_holds_when_active(monkeypatch, tmp_path):
    monkeypatch.setenv("SOPHIA_MCP_APPROVAL", "1")
    from sophia_mcp import approval
    assert approval.requires_approval("sophia_wiki_upsert") is True
    assert approval.requires_approval("sophia_check_claim") is False  # not on the list

    q = tmp_path / "queue.jsonl"
    held = approval.enqueue("sophia_wiki_upsert",
                            {"title": "X", "api_key": "super-secret-value"},
                            role="agent", queue_path=q)
    assert held["result"] is None
    assert held["_governance"]["verdict"] == "held"
    assert held["held_reason"] == "approval_required"
    # secret-safety: the raw value is NEVER written to the queue
    body = q.read_text(encoding="utf-8")
    assert "super-secret-value" not in body
    assert "api_key" in body  # key NAMES are fine; values are not
    assert approval.pending(q)[0]["tool"] == "sophia_wiki_upsert"


def test_gateway_governed_does_not_dispatch_when_approval_required(monkeypatch):
    """The served path must HOLD (no side effect) before reaching the handler."""
    monkeypatch.setenv("SOPHIA_MCP_APPROVAL", "1")
    from sophia_mcp import gateway_wiring, approval

    gateway_wiring.reset()
    monkeypatch.setattr(approval, "APPROVAL_QUEUE", __import__("pathlib").Path("/tmp/_sophiark_test_queue.jsonl"))
    # export_corpus is on the approval list AND governed; with the gate on it must hold.
    out = gateway_wiring.governed("sophia_export_corpus", {})
    assert out.get("result") is None
    assert out.get("_governance", {}).get("held_reason") == "approval_required"
    gateway_wiring.reset()


# --------------------------------------------------------------------------- #
# 4. sentinel_watch skill is fail-closed
# --------------------------------------------------------------------------- #

def test_sentinel_watch_skill_flags_review_on_gate_failure():
    out = run_skill("sentinel_watch",
                    step={"type": "step_output", "passed": False, "failureClass": "gate_violation"})
    assert out["ok"] is True and out["verdict"] == "review"
    assert out["failureClass"] == "gate_violation"


def test_sentinel_watch_skill_ok_on_clean_step():
    out = run_skill("sentinel_watch", step={"type": "step_output", "passed": True})
    assert out["verdict"] == "ok"


def test_sentinel_watch_skill_never_raises_on_bad_input():
    out = run_skill("sentinel_watch", step="not a dict")
    assert isinstance(out, dict) and out["ok"] is True  # wrapper kept it safe

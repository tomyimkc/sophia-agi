# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the fail-closed gated auto-remediation (R4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster import ledger as L  # noqa: E402
from agent.cluster.heal import Decision, RemediationGate, plan_remediation  # noqa: E402
from agent.cluster.playbook import Diagnosis, Risk  # noqa: E402


def _diag(action="gc_disk", risk=Risk.LOW, conf=0.9):
    return Diagnosis(signal="disk_used_frac", root_cause="disk full", remediation="gc",
                     action=action, risk=risk, confidence=conf)


def test_low_risk_high_conf_safe_action_auto_heals(tmp_path):
    gate = RemediationGate()
    decision, _ = gate.decide(_diag())
    assert decision == Decision.AUTO_HEAL


def test_high_risk_escalates():
    decision, reason = RemediationGate().decide(_diag(action="drain_and_diag", risk=Risk.HIGH))
    assert decision == Decision.ESCALATE
    assert "risk" in reason


def test_low_confidence_escalates():
    decision, _ = RemediationGate().decide(_diag(conf=0.5))
    assert decision == Decision.ESCALATE


def test_action_not_on_allowlist_escalates():
    # LOW risk + high conf but a non-allowlisted action must still escalate.
    decision, _ = RemediationGate().decide(_diag(action="drain_and_reboot", risk=Risk.LOW))
    assert decision == Decision.ESCALATE


def test_observe_action_is_observe():
    decision, _ = RemediationGate().decide(_diag(action="observe"))
    assert decision == Decision.OBSERVE


def test_dry_run_never_executes_without_approval(tmp_path):
    audit = tmp_path / "audit.jsonl"
    executed = []
    plan = plan_remediation("n0", _diag(), executor=lambda n, a: executed.append((n, a)) or True,
                            allow=False, audit_path=audit)
    assert plan.decision == Decision.AUTO_HEAL
    assert plan.dry_run and not plan.executed
    assert executed == []  # executor never called without approval
    # decision still audited
    rec = json.loads(audit.read_text().splitlines()[-1])
    assert rec["tool"] == "cluster_heal" and rec["executed"] is False


def test_approved_auto_heal_executes_and_records_recovery(tmp_path):
    audit = tmp_path / "audit.jsonl"
    led = tmp_path / "incidents.jsonl"
    iid = L.record_detection(node_id="n0", kind="disk_used_frac", severity="WARN",
                             detected_at="2026-06-26T00:00:00+00:00", ledger=led)
    plan = plan_remediation("n0", _diag(), executor=lambda n, a: True, allow=True,
                            audit_path=audit, incident_id=iid, ledger=led)
    assert plan.executed and not plan.dry_run
    stats = L.mttr_stats(led)
    assert stats["recovered"] == 1 and stats["auto_healed"] == 1


def test_escalation_records_to_ledger(tmp_path):
    audit = tmp_path / "audit.jsonl"
    led = tmp_path / "incidents.jsonl"
    iid = L.record_detection(node_id="n0", kind="ecc_uncorrectable", severity="FAIL",
                             detected_at="2026-06-26T00:00:00+00:00", ledger=led)
    plan = plan_remediation("n0", _diag(action="drain_and_diag", risk=Risk.HIGH),
                            allow=True, audit_path=audit, incident_id=iid, ledger=led)
    assert plan.decision == Decision.ESCALATE
    assert L.mttr_stats(led)["escalated"] == 1

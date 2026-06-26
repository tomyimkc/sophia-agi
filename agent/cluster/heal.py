# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gated auto-remediation (R4: 自动化运维 / 提升故障自愈率).

The contribution here is *not* "reboot the node automatically". It is a fail-closed,
risk-proportional **decision gate** between a diagnosis and an action — VISION.md's
"no autonomous high-stakes action without oversight" applied to ops:

    diagnosis ──▶ RemediationGate.decide ──▶ AUTO_HEAL | ESCALATE | OBSERVE
                       │
                       ├─ risk is LOW   and confidence ≥ auto_threshold  ▶ AUTO_HEAL
                       ├─ action is observe-only                         ▶ OBSERVE
                       └─ otherwise (MEDIUM/HIGH risk, or low confidence) ▶ ESCALATE

Every decision is appended to the MCP audit log (``sophia_mcp.audit``) and reflected
in the incident ledger, so the **self-heal ratio** (auto vs. escalated) is measured
and the 自愈率 improvement is evidence-backed, not asserted.

Execution is *dry-run by default*. A real action only runs when (a) the gate says
AUTO_HEAL, (b) the operator opted in via ``SOPHIA_CLUSTER_HEAL=1`` (or ``allow=True``),
and (c) an executor is provided. With no executor, the planner emits the intended
command and records it without touching any node — exactly like the repo's
``--dry-run`` tools.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from agent.cluster import ledger as ledger_mod
from agent.cluster.playbook import Diagnosis, Risk
from sophia_mcp.audit import audit_log

APPROVE_ENV = "SOPHIA_CLUSTER_HEAL"  # set to "1" to allow real (non-dry-run) actions

# Actions safe enough to auto-execute when confidence is high. Deliberately a tiny,
# explicit allowlist of non-destructive operations — drains, reboots and RMAs are
# never on it.
AUTO_SAFE_ACTIONS: frozenset[str] = frozenset({
    "gc_disk",            # delete old checkpoints/logs to relieve disk pressure
    "restore_telemetry",  # restart the DCGM/nvidia-smi agent
    "observe",            # capture diagnostics only
})


class Decision:
    AUTO_HEAL = "auto_heal"
    ESCALATE = "escalate"
    OBSERVE = "observe"


@dataclass(frozen=True)
class RemediationPlan:
    node_id: str
    decision: str
    action: str
    risk: str
    confidence: float
    reason: str
    diagnosis: Diagnosis
    executed: bool = False
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "decision": self.decision,
            "action": self.action,
            "risk": self.risk,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "executed": self.executed,
            "dry_run": self.dry_run,
            "diagnosis": self.diagnosis.to_dict(),
        }


@dataclass(frozen=True)
class RemediationGate:
    """The fail-closed decision boundary. Tunable but conservative by default."""

    auto_confidence: float = 0.75

    def decide(self, diag: Diagnosis) -> tuple[str, str]:
        """Return (decision, reason). Fail-closed toward ESCALATE."""

        if diag.action == "observe":
            return Decision.OBSERVE, "observe-only diagnosis; capture diagnostics, no action"
        if diag.risk != Risk.LOW:
            return Decision.ESCALATE, f"risk={diag.risk} exceeds auto threshold (LOW); human-in-the-loop"
        if diag.action not in AUTO_SAFE_ACTIONS:
            return Decision.ESCALATE, f"action '{diag.action}' not on the auto-safe allowlist"
        if diag.confidence < self.auto_confidence:
            return (Decision.ESCALATE,
                    f"confidence {diag.confidence:.2f} < auto threshold {self.auto_confidence:.2f}")
        return Decision.AUTO_HEAL, (
            f"LOW risk, confidence {diag.confidence:.2f} ≥ {self.auto_confidence:.2f}, "
            f"action on auto-safe allowlist"
        )


def plan_remediation(
    node_id: str,
    diag: Diagnosis,
    *,
    gate: RemediationGate | None = None,
    executor: Callable[[str, str], bool] | None = None,
    allow: bool | None = None,
    audit_path: Path | None = None,
    incident_id: str | None = None,
    ledger: Path = ledger_mod.DEFAULT_LEDGER,
) -> RemediationPlan:
    """Gate a diagnosis into a remediation plan; optionally execute if AUTO_HEAL.

    ``allow`` (or ``SOPHIA_CLUSTER_HEAL=1``) must be true for a real action to run; an
    ``executor`` must also be supplied. Otherwise the plan is computed and recorded but
    nothing touches the node (dry-run).
    """

    gate = gate or RemediationGate()
    decision, reason = gate.decide(diag)

    approved = allow if allow is not None else (os.environ.get(APPROVE_ENV) == "1")
    executed = False
    dry_run = True

    if decision == Decision.AUTO_HEAL and approved and executor is not None:
        dry_run = False
        executed = bool(executor(node_id, diag.action))

    plan = RemediationPlan(
        node_id=node_id, decision=decision, action=diag.action, risk=diag.risk,
        confidence=diag.confidence, reason=reason, diagnosis=diag,
        executed=executed, dry_run=dry_run,
    )

    # Audit every decision (defense-in-depth: even dry-run decisions are logged).
    audit_entry = {
        "tool": "cluster_heal", "node_id": node_id, "decision": decision,
        "action": diag.action, "risk": diag.risk, "confidence": round(diag.confidence, 3),
        "executed": executed, "dry_run": dry_run, "reason": reason,
    }
    if audit_path is not None:
        audit_log(audit_entry, path=audit_path)
    else:
        audit_log(audit_entry)

    # Reflect the outcome in the incident ledger so MTTR/self-heal ratio stay measured.
    if incident_id is not None:
        if decision == Decision.ESCALATE:
            ledger_mod.record_escalation(incident_id, reason=reason, ledger=ledger)
        elif decision == Decision.AUTO_HEAL and executed:
            ledger_mod.record_recovery(incident_id, auto_healed=True,
                                       note=f"auto-healed via {diag.action}", ledger=ledger)

    return plan

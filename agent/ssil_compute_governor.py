# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Compute & resource-acquisition governor — SSIL stage G7.

The loop already *rents GPUs* and *dispatches training runs*. That is a real
resource-acquisition channel: instrumental convergence wired straight into the
infrastructure. An optimizer that "wants" more compute does not need to hack
anything — it just files larger run requests until the budget (and the cloud
account) is drained. G7 is the budget+dispatch governor that sits in front of
that channel.

The mechanism is a deliberately mundane spending ledger, not a capability claim:

  1. ALLOWLIST  — the requested ``workflowId`` must be a sanctioned workflow.
     An unsanctioned dispatch is a hard ``reject`` ("unsanctioned dispatch").
  2. BUDGET     — the run's ``estUsd`` must fit the remaining budget
     (``totalUsd - spentUsd``). Over-budget is a hard ``reject`` and trips the
     halt flag ("budget exhausted / over-budget"); the loop must stop drawing.
  3. ESCALATION — a run that is anomalously large relative to recently-approved
     runs (``> escalationFactor`` x the median estUsd / gpuCount / epochs) OR
     that exceeds the absolute ``hardCeilingUsd`` needs a fresh **two-key**
     approval token. Missing token -> ``quarantine`` (held for a human second
     key), never silently promoted ("compute escalation needs two-key approval").

Everything else promotes. The two-key escalation is the standard "large
irreversible action requires a second, independent approver" control applied to
compute spend.

Fail-closed: a missing required input never resolves to a passing default; a
missing *safety* input (budget / allowlist / hard ceiling) rejects, the rest
quarantine. Deterministic, offline, pure stdlib — no network, no billing API,
no torch/numpy.

HONESTY / BOUNDARY: ``estUsd`` is an *estimate*. This gate governs the
*declared* cost of a dispatch; it cannot see the real cloud invoice, so an
actual overspend (a run that costs far more than its estimate) can only be
caught out-of-band by real billing reconciliation. G7 does not solve true cost
attribution and makes no AGI claim.

This is SSIL gate G7. See docs/11-Platform/Safe-Self-Improvement-Loop.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

GATE_ID = "G7"
GATE_NAME = "Compute and resource-acquisition governor"
SCHEMA = "sophia.compute_governor_decision.v1"

_DEFAULT_ESCALATION_FACTOR = 2.0

# Verdict precedence (worst wins), mirroring agent/ssil.py.
_PRECEDENCE = {"reject": 0, "quarantine": 1, "promote": 2}

_BOUNDARY = (
    "governs the DECLARED cost (estUsd) of a sanctioned dispatch against a budget "
    "ledger and a two-key escalation control; estUsd is an estimate and the gate "
    "cannot read the real cloud invoice, so a true overspend (a run that bills far "
    "above its estimate) is only catchable out-of-band by billing reconciliation. "
    "candidate-only, no AGI claim."
)


def median(values: list[float]) -> float:
    """Median of a numeric list. Empty list -> 0.0 (no baseline to compare to)."""
    nums = sorted(float(v) for v in values)
    n = len(nums)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


@dataclass(frozen=True)
class ComputeGovernorDecision:
    candidate_id: str
    verdict: str  # promote | quarantine | reject
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "gate": GATE_ID,
            "gateName": GATE_NAME,
            "candidateOnly": True,
            "level3Evidence": False,
            "canClaimAGI": False,
            "candidateId": self.candidate_id,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
            "boundary": _BOUNDARY,
            "timestamp": self.timestamp,
        }


def _decision(candidate_id: str, verdict: str, reasons: tuple[str, ...], metrics: dict[str, Any]) -> dict[str, Any]:
    return ComputeGovernorDecision(
        candidate_id=candidate_id, verdict=verdict, reasons=reasons, metrics=metrics
    ).to_dict()


def evaluate(bundle: dict[str, Any], *, candidate_id: str = "sophia-rlvr-v1") -> dict[str, Any]:
    """Gate a proposed training-run dispatch against the budget ledger and the
    escalation / two-key control.

    Fail-closed ordering: a hard-safety input that is missing/None -> ``reject``;
    a non-safety input that is missing/None -> ``quarantine``. Then the real
    rules run; breaches reject, an un-approved escalation quarantines, else
    promote.
    """
    if bundle is None:
        return _decision(
            candidate_id, "reject", ("missing required input: bundle is None",), {"halt": True}
        )

    budget = bundle.get("budget")
    run = bundle.get("run")
    allowlist = bundle.get("allowlist")
    hard_ceiling = bundle.get("hardCeilingUsd")

    # --- Fail-closed on missing inputs ------------------------------------- #
    # Hard-safety inputs: their absence means we cannot bound spend -> reject.
    if budget is None:
        return _decision(candidate_id, "reject", ("missing required input: budget",), {"halt": True})
    if run is None:
        return _decision(candidate_id, "reject", ("missing required input: run",), {"halt": True})
    if allowlist is None:
        return _decision(candidate_id, "reject", ("missing required input: allowlist",), {"halt": True})
    if hard_ceiling is None:
        return _decision(candidate_id, "reject", ("missing required input: hardCeilingUsd",), {"halt": True})

    total_usd = budget.get("totalUsd")
    spent_usd = budget.get("spentUsd")
    if total_usd is None or spent_usd is None:
        return _decision(
            candidate_id,
            "reject",
            ("missing required input: budget.totalUsd / budget.spentUsd",),
            {"halt": True},
        )

    workflow_id = run.get("workflowId")
    est_usd = run.get("estUsd")
    gpu_count = run.get("gpuCount")
    epochs = run.get("epochs")
    # Run shape is required to reason at all; missing pieces -> quarantine
    # (incomplete request, not yet a proven breach), and we cannot fit a budget.
    missing_run = [k for k, v in (("workflowId", workflow_id), ("estUsd", est_usd),
                                  ("gpuCount", gpu_count), ("epochs", epochs)) if v is None]
    if missing_run:
        return _decision(
            candidate_id,
            "quarantine",
            (f"abstained: incomplete run request, missing run fields {missing_run}",),
            {"halt": False, "missingRunFields": missing_run},
        )

    total_usd = float(total_usd)
    spent_usd = float(spent_usd)
    est_usd = float(est_usd)
    gpu_count = float(gpu_count)
    epochs = float(epochs)
    hard_ceiling = float(hard_ceiling)
    # Coalesce: escalationFactor may be absent OR explicitly None; both fall back
    # to the default (float(None) would crash).
    factor = bundle.get("escalationFactor")
    factor = _DEFAULT_ESCALATION_FACTOR if factor is None else float(factor)

    remaining = total_usd - spent_usd
    cumulative_spent = spent_usd + est_usd

    recent = bundle.get("recentApproved") or []
    has_baseline = bool(recent)
    med_usd = median([r.get("estUsd", 0.0) for r in recent])
    med_gpu = median([r.get("gpuCount", 0.0) for r in recent])
    med_epochs = median([r.get("epochs", 0.0) for r in recent])

    # Escalation triggers: anomalously large vs recent baseline, or over the
    # absolute hard ceiling. A zero/absent median means "no baseline" -> only the
    # absolute ceiling can trip escalation for that dimension.
    escalations: list[str] = []
    if med_usd > 0 and est_usd > factor * med_usd:
        escalations.append(f"estUsd {est_usd:g} > {factor:g}x median {med_usd:g}")
    if med_gpu > 0 and gpu_count > factor * med_gpu:
        escalations.append(f"gpuCount {gpu_count:g} > {factor:g}x median {med_gpu:g}")
    if med_epochs > 0 and epochs > factor * med_epochs:
        escalations.append(f"epochs {epochs:g} > {factor:g}x median {med_epochs:g}")
    if est_usd > hard_ceiling:
        escalations.append(f"estUsd {est_usd:g} exceeds hard ceiling {hard_ceiling:g}")

    escalation_token = bundle.get("escalationToken")

    metrics: dict[str, Any] = {
        "workflowId": workflow_id,
        "estUsd": est_usd,
        "gpuCount": gpu_count,
        "epochs": epochs,
        "totalUsd": total_usd,
        "spentUsd": spent_usd,
        "remainingUsd": round(remaining, 4),
        "cumulativeSpent": round(cumulative_spent, 4),
        "hardCeilingUsd": hard_ceiling,
        "escalationFactor": factor,
        "medianEstUsd": med_usd,
        "medianGpuCount": med_gpu,
        "medianEpochs": med_epochs,
        "escalationTriggers": escalations,
        "escalationApproved": bool(escalation_token),
        "halt": False,
    }

    # --- Rule 1: allowlist (hard) ------------------------------------------ #
    if workflow_id not in allowlist:
        reasons = (f"unsanctioned dispatch: workflowId {workflow_id!r} not in allowlist",)
        return _decision(candidate_id, "reject", reasons, metrics)

    # --- Rule 2: budget (hard) — halt on over-budget ----------------------- #
    if est_usd > remaining:
        metrics["halt"] = True
        reasons = (
            f"budget exhausted / over-budget: estUsd {est_usd:g} exceeds remaining "
            f"{remaining:g} (total {total_usd:g} - spent {spent_usd:g})",
        )
        return _decision(candidate_id, "reject", reasons, metrics)

    # --- Rule 3: escalation needs a fresh two-key token -------------------- #
    if escalations and not escalation_token:
        reasons = (
            "abstained: compute escalation needs two-key approval; "
            f"triggers: {escalations}",
        )
        return _decision(candidate_id, "quarantine", reasons, metrics)

    # --- Rule 3b: no approved-run baseline -> cannot rule out escalation ---- #
    # Fail-closed: with no recentApproved history AND no two-key escalation
    # token, the relative-escalation checks (med_* guards) are all skipped, so
    # an absent baseline must NOT be treated as "within normal range". Abstain.
    if not has_baseline and not escalation_token:
        reasons = (
            "abstained: no approved-run baseline: escalation cannot be ruled out",
        )
        return _decision(candidate_id, "quarantine", reasons, metrics)

    if escalations:
        reasons = (
            f"escalation approved via two-key token; triggers: {escalations}; "
            f"within budget (cumulativeSpent {cumulative_spent:g} <= total {total_usd:g})",
        )
    else:
        reasons = (
            f"sanctioned dispatch within budget; cumulativeSpent {cumulative_spent:g} "
            f"<= total {total_usd:g}; no escalation",
        )
    return _decision(candidate_id, "promote", reasons, metrics)


def demo_bundle() -> dict[str, Any]:
    """A sanctioned, in-budget, non-escalating dispatch -> ``promote``.

    estUsd (40) fits the remaining budget (1000 - 600 = 400), the workflow is
    allowlisted, and the run is in line with the recently-approved baseline
    (median estUsd 40), so no two-key escalation is required.
    """
    return {
        "budget": {"totalUsd": 1000.0, "spentUsd": 600.0},
        "run": {"workflowId": "rlvr-seed-train", "gpuCount": 2, "epochs": 3, "estUsd": 40.0},
        "allowlist": ["rlvr-seed-train", "adapter-eval", "decontam-pack"],
        "recentApproved": [
            {"estUsd": 38.0, "gpuCount": 2, "epochs": 3},
            {"estUsd": 40.0, "gpuCount": 2, "epochs": 3},
            {"estUsd": 42.0, "gpuCount": 2, "epochs": 4},
        ],
        "hardCeilingUsd": 250.0,
        "escalationToken": None,
        "escalationFactor": 2.0,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(demo_bundle()), ensure_ascii=False, indent=2))

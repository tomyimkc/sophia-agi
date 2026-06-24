# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Uncertainty-driven active verification daemon.

Sophia should not go silent on out-of-wiki claims. This module turns holds,
low-confidence accepts, stale provenance, and calibration gaps into a prioritized
agenda of *active* verification work. It emits plans/candidates; canonical wiki
promotion remains gated elsewhere.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent.fact_check_gate import decision_to_dict, fact_check_text


@dataclass(frozen=True)
class ActiveGap:
    id: str
    claim: str
    claim_type: str = "open_empirical"
    risk: str = "normal"
    reason: str = "held"
    confidence: float = 0.0
    source: str = "report"
    priority: float = 0.0
    signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationAction:
    kind: str
    target: str
    cost: float = 1.0
    expected_gain: float = 0.0
    offline: bool = True


@dataclass(frozen=True)
class ActivePlan:
    gap: ActiveGap
    actions: tuple[VerificationAction, ...]
    expected_information_gain: float
    decision_rule: str = "re-run fact_check_text; accept only accepted; otherwise keep held/quarantine"

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap": self.gap.__dict__ | {"signals": list(self.gap.signals)},
            "actions": [a.__dict__ for a in self.actions],
            "expectedInformationGain": self.expected_information_gain,
            "decisionRule": self.decision_rule,
        }


def _case_claims(case: dict[str, Any]) -> list[dict[str, Any]]:
    return list(case.get("claims") or []) or [{
        "claim": case.get("claim", ""),
        "type": case.get("type", "open_empirical"),
        "risk": case.get("risk", "normal"),
        "verdict": case.get("verdict", "held"),
        "confidence": case.get("confidence", 0.0),
        "reason": case.get("reason", ""),
    }]


def discover_gaps(report: dict[str, Any], *, low_conf_floor: float = 0.82) -> list[ActiveGap]:
    """Extract active-learning gaps from a fact-check/eval report."""
    gaps: list[ActiveGap] = []
    for i, case in enumerate(report.get("cases", [])):
        for j, c in enumerate(_case_claims(case)):
            verdict = c.get("verdict", case.get("verdict", "held"))
            conf = float(c.get("confidence", case.get("confidence", 0.0)) or 0.0)
            signals: list[str] = []
            if verdict == "held":
                signals.append("held")
            if verdict == "accepted" and conf < low_conf_floor:
                signals.append("low_confidence_accept")
            if "insufficient" in str(c.get("reason", "")).lower():
                signals.append("insufficient_sources")
            if c.get("learningCandidate"):
                signals.append("candidate_needs_recheck")
            if not signals:
                continue
            risk = str(c.get("risk", case.get("risk", "normal")))
            weight = 2.0 if risk == "high" else 1.0
            priority = weight * (1.0 - min(conf, 0.99)) + 0.35 * signals.count("insufficient_sources") + 0.25 * signals.count("candidate_needs_recheck")
            gaps.append(ActiveGap(
                id=f"{case.get('id', 'case'+str(i))}:{j}",
                claim=str(c.get("claim", case.get("claim", ""))),
                claim_type=str(c.get("type", case.get("type", "open_empirical"))),
                risk=risk,
                reason=str(c.get("reason", case.get("reason", ""))),
                confidence=round(conf, 4),
                source=str(case.get("id", "report")),
                priority=round(priority, 4),
                signals=tuple(signals),
            ))
    return sorted(gaps, key=lambda g: (g.priority, g.risk == "high"), reverse=True)


def plan_for_gap(gap: ActiveGap) -> ActivePlan:
    """Generate a concrete, pluggable verification plan for one gap."""
    actions: list[VerificationAction] = []
    t = gap.claim_type
    high = gap.risk == "high"
    if t == "doi":
        actions.append(VerificationAction("resolve_doi", gap.claim, cost=0.2, expected_gain=0.85, offline=False))
        actions.append(VerificationAction("crossref_metadata", gap.claim, cost=0.4, expected_gain=0.75, offline=False))
    elif t == "url":
        actions.append(VerificationAction("resolve_url", gap.claim, cost=0.2, expected_gain=0.75, offline=False))
    elif t.startswith("econ"):
        actions.extend([
            VerificationAction("world_bank", gap.claim, cost=0.5, expected_gain=0.72, offline=False),
            VerificationAction("fred_or_bls", gap.claim, cost=0.6, expected_gain=0.68, offline=False),
            VerificationAction("independent_macro_source", gap.claim, cost=0.8, expected_gain=0.55, offline=False),
        ])
    elif "authorship" in t or "open" in t:
        actions.extend([
            VerificationAction("wikidata", gap.claim, cost=0.4, expected_gain=0.7, offline=False),
            VerificationAction("crossref_or_openalex", gap.claim, cost=0.5, expected_gain=0.62, offline=False),
            VerificationAction("source_family_check", gap.claim, cost=0.2, expected_gain=0.35, offline=True),
        ])
    else:
        actions.extend([
            VerificationAction("deterministic_type_probe", gap.claim, cost=0.2, expected_gain=0.35, offline=True),
            VerificationAction("synthesize_verifier_candidate", gap.claim, cost=0.8, expected_gain=0.55, offline=True),
            VerificationAction("retrieval_search", gap.claim, cost=0.7, expected_gain=0.5, offline=False),
        ])
    if high:
        actions.append(VerificationAction("third_independent_source_required", gap.claim, cost=0.5, expected_gain=0.45, offline=False))
    eig = sum(a.expected_gain / max(a.cost, 0.1) for a in actions)
    # Diminishing returns + uncertainty weight.
    eig = (1 - math.exp(-eig / 4.0)) * (1.2 if high else 1.0) * (1.0 - gap.confidence / 2.0)
    return ActivePlan(gap=gap, actions=tuple(actions), expected_information_gain=round(eig, 4))


def build_active_agenda(report: dict[str, Any], *, limit: int = 20) -> dict[str, Any]:
    gaps = discover_gaps(report)[:limit]
    plans = [plan_for_gap(g).to_dict() for g in gaps]
    return {
        "schema": "sophia.active_inference_agenda.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "nGaps": len(gaps),
        "plans": plans,
        "invariants": {
            "all_gaps_have_actions": all(p["actions"] for p in plans),
            "high_risk_gets_extra_source": all(
                any(a["kind"] == "third_independent_source_required" for a in p["actions"])
                for p in plans if p["gap"].get("risk") == "high"
            ),
            "no_canonical_write": True,
        },
    }


def run_active_cycle(report: dict[str, Any], *, verifier_kwargs: dict[str, Any] | None = None, limit: int = 10) -> dict[str, Any]:
    """Try to close top gaps by re-running the fact-check gate with injected backends.

    This remains fail-closed: accepted decisions become learning candidates;
    held/rejected decisions stay non-published.
    """
    verifier_kwargs = verifier_kwargs or {}
    agenda = build_active_agenda(report, limit=limit)
    outcomes = []
    for p in agenda["plans"]:
        claim = p["gap"]["claim"]
        dec = decision_to_dict(fact_check_text(claim, **verifier_kwargs))
        outcomes.append({
            "gapId": p["gap"]["id"],
            "claim": claim,
            "verdict": dec["verdict"],
            "reason": dec["reason"],
            "learningCandidates": [c.get("learningCandidate") for c in dec.get("claims", []) if c.get("learningCandidate")],
        })
    accepted = sum(1 for o in outcomes if o["verdict"] == "accepted")
    return {
        "schema": "sophia.active_inference_cycle.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "agenda": agenda,
        "outcomes": outcomes,
        "metrics": {
            "attempted": len(outcomes),
            "acceptedForQuarantine": accepted,
            "stillFailClosed": sum(1 for o in outcomes if o["verdict"] != "accepted"),
        },
    }


def write_active_agenda(report_path: str | Path, out: str | Path, *, limit: int = 20) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    agenda = build_active_agenda(report, limit=limit)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(agenda, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return agenda


__all__ = ["ActiveGap", "VerificationAction", "ActivePlan", "discover_gaps", "plan_for_gap", "build_active_agenda", "run_active_cycle", "write_active_agenda"]

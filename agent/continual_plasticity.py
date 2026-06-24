"""Safe continual-plasticity gate for model/skill updates.

No weights are changed here. The module decides whether a proposed adapter/skill
update may be promoted based on held-out improvement, protected-suite regression,
contamination checks, and verifier artifacts. This is the infrastructure that a
future LoRA/RLVR job must clear before it affects Sophia.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.conscience_enforcement import enforce_conscience


@dataclass(frozen=True)
class EvalMetric:
    suite: str
    before: float
    after: float
    protected: bool = False

    @property
    def delta(self) -> float:
        return round(float(self.after) - float(self.before), 4)


@dataclass(frozen=True)
class UpdateCandidate:
    id: str
    kind: str
    metrics: tuple[EvalMetric, ...]
    verifier_artifacts: tuple[str, ...] = ()
    contaminated: bool = False
    notes: str = ""


@dataclass(frozen=True)
class PromotionDecision:
    candidate_id: str
    verdict: str  # promote|quarantine|reject
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "sophia.continual_plasticity_decision.v1",
            "candidateOnly": True,
            "level3Evidence": False,
            "candidateId": self.candidate_id,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


def evaluate_update(candidate: UpdateCandidate, *, target_suite: str, min_target_delta: float = 0.03, max_protected_regression: float = 0.01, require_artifacts: int = 2) -> PromotionDecision:
    reasons: list[str] = []
    conscience = enforce_conscience(
        action="train_or_update_adapter",
        text=f"{candidate.kind} {candidate.id} {candidate.notes}",
        context={"externalApproval": False, "allowCautionVerdicts": True},
        high_impact=True,
    )
    if not conscience.allowed and conscience.verdict == "block":
        reasons.append(f"conscience blocked update: {conscience.reason}")
    if candidate.contaminated:
        reasons.append("contamination flag set")
    if len(candidate.verifier_artifacts) < require_artifacts:
        reasons.append(f"insufficient verifier artifacts: {len(candidate.verifier_artifacts)}/{require_artifacts}")
    target = [m for m in candidate.metrics if m.suite == target_suite]
    if not target:
        reasons.append(f"missing target suite: {target_suite}")
        target_delta = 0.0
    else:
        target_delta = target[0].delta
        if target_delta < min_target_delta:
            reasons.append(f"target improvement below floor: {target_delta:.4f} < {min_target_delta:.4f}")
    regressions = [m for m in candidate.metrics if m.protected and m.delta < -max_protected_regression]
    for m in regressions:
        reasons.append(f"protected regression on {m.suite}: {m.delta:.4f}")
    if candidate.contaminated or regressions:
        verdict = "reject"
    elif reasons:
        verdict = "quarantine"
    else:
        verdict = "promote"
    return PromotionDecision(
        candidate_id=candidate.id,
        verdict=verdict,
        reasons=tuple(reasons or ["all gates cleared"]),
        metrics={
            "targetSuite": target_suite,
            "targetDelta": round(target_delta, 4),
            "maxProtectedRegression": max([m.delta for m in candidate.metrics if m.protected] or [0.0]),
            "artifactCount": len(candidate.verifier_artifacts),
            "metricRows": [m.__dict__ | {"delta": m.delta} for m in candidate.metrics],
        },
    )


def append_promotion_ledger(decision: PromotionDecision, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")


def demo_plasticity_report() -> dict[str, Any]:
    good = UpdateCandidate(
        id="adapter_router_v1",
        kind="lora_adapter",
        verifier_artifacts=("fact-check-heldout", "provenance-delta"),
        metrics=(
            EvalMetric("tool_routing", 0.72, 0.79, protected=False),
            EvalMetric("source_discipline", 0.98, 0.98, protected=True),
            EvalMetric("fact_check_false_accept", 0.99, 0.99, protected=True),
        ),
    )
    bad = UpdateCandidate(
        id="adapter_overfit_v1",
        kind="lora_adapter",
        verifier_artifacts=("train-score",),
        contaminated=True,
        metrics=(EvalMetric("tool_routing", 0.72, 0.9), EvalMetric("source_discipline", 0.98, 0.92, protected=True)),
    )
    decisions = [evaluate_update(good, target_suite="tool_routing"), evaluate_update(bad, target_suite="tool_routing")]
    return {
        "schema": "sophia.continual_plasticity_demo.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "decisions": [d.to_dict() for d in decisions],
        "invariants": {
            "clean_improving_update_promotes": decisions[0].verdict == "promote",
            "contaminated_or_regressing_update_rejects": decisions[1].verdict == "reject",
        },
    }


def write_plasticity_report(out: str | Path) -> dict[str, Any]:
    report = demo_plasticity_report()
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


__all__ = ["EvalMetric", "UpdateCandidate", "PromotionDecision", "evaluate_update", "append_promotion_ledger", "demo_plasticity_report", "write_plasticity_report"]

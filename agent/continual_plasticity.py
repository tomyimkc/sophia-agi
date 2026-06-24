# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
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
class RetentionEvidence:
    """Old-task retention signal from a learning-under-shift run.

    `old_benchmark_delta_pct` is (post-learning old-benchmark score − baseline), in
    percentage points; a negative value means the update degraded previously-learned
    knowledge (catastrophic forgetting). `evaluable` mirrors the learning-shift
    `stabilityEvaluable` field: "evaluated" | "not-requested" | "requested-but-no-baseline".
    Only an `evaluated` signal can prove or disprove forgetting; anything else is
    unverifiable and must not be treated as a silent pass.
    """

    old_benchmark_delta_pct: float | None
    passing_signal: bool | None = None
    evaluable: str = "evaluated"
    source: str = ""

    @property
    def verifiable(self) -> bool:
        return self.evaluable == "evaluated" and self.old_benchmark_delta_pct is not None

    def forgot(self, max_regression_pct: float) -> bool:
        """True iff retention is verifiable AND the old task regressed beyond tolerance."""
        return self.verifiable and float(self.old_benchmark_delta_pct) < -abs(max_regression_pct)


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


@dataclass(frozen=True)
class Goal:
    """One objective in a multi-goal promotion: a suite that must improve by at least
    `min_delta`. A goal that only needs to hold steady uses `min_delta=0.0`."""

    suite: str
    min_delta: float = 0.03


def _conscience_block_reason(candidate: UpdateCandidate) -> str | None:
    conscience = enforce_conscience(
        action="train_or_update_adapter",
        text=f"{candidate.kind} {candidate.id} {candidate.notes}",
        context={"externalApproval": False, "allowCautionVerdicts": True},
        high_impact=True,
    )
    if not conscience.allowed and conscience.verdict == "block":
        return f"conscience blocked update: {conscience.reason}"
    return None


def _retention_reasons(retention: RetentionEvidence | None, *, max_regression_pct: float, required: bool) -> tuple[list[str], bool]:
    """Shared retention checks. Returns (reasons, forgetting). `forgetting` is a hard reject."""
    reasons: list[str] = []
    forgetting = retention is not None and retention.forgot(max_regression_pct)
    if forgetting:
        reasons.append(
            f"old-task retention regression: {retention.old_benchmark_delta_pct:.2f}pp "
            f"< -{abs(max_regression_pct):.2f}pp (catastrophic forgetting)"
        )
    if required and (retention is None or not retention.verifiable):
        reasons.append("retention evidence required but not verifiable (no learning-under-shift baseline)")
    return reasons, forgetting


def _retention_metrics(retention: RetentionEvidence | None, *, max_regression_pct: float, required: bool, forgetting: bool) -> dict[str, Any]:
    return {
        "oldBenchmarkDeltaPct": retention.old_benchmark_delta_pct if retention else None,
        "passingSignal": retention.passing_signal if retention else None,
        "evaluable": retention.evaluable if retention else "not-provided",
        "maxRetentionRegressionPct": max_regression_pct,
        "required": required,
        "forgetting": forgetting,
    }


def evaluate_update_multigoal(candidate: UpdateCandidate, *, goals: tuple[Goal, ...], max_regression: float = 0.01, require_artifacts: int = 2, retention: RetentionEvidence | None = None, max_retention_regression_pct: float = 5.0, require_retention: bool = False) -> PromotionDecision:
    """Promote only on a Pareto improvement across ALL goals.

    Generalizes `evaluate_update` from one target suite to N. Lifting one goal by
    sacrificing another is the multi-goal failure mode (v3 improved aggregate while
    religion stayed flat), so the rules are:

      - HARD REJECT if the dataset is contaminated, retention shows catastrophic
        forgetting, or ANY goal / protected suite regresses below `-max_regression`
        (a cross-goal trade-off).
      - QUARANTINE (fail-closed abstain) if any goal misses its `min_delta` floor, a
        goal suite is unmeasured, artifacts are insufficient, conscience cautions, or
        required retention is unverifiable.
      - PROMOTE only when every goal clears its floor with no regression anywhere.
    """
    if not goals:
        raise ValueError("evaluate_update_multigoal requires at least one goal")
    by_suite = {m.suite: m for m in candidate.metrics}
    reasons: list[str] = []

    block = _conscience_block_reason(candidate)
    if block:
        reasons.append(block)
    if candidate.contaminated:
        reasons.append("contamination flag set")
    if len(candidate.verifier_artifacts) < require_artifacts:
        reasons.append(f"insufficient verifier artifacts: {len(candidate.verifier_artifacts)}/{require_artifacts}")

    missing = [g.suite for g in goals if g.suite not in by_suite]
    for s in missing:
        reasons.append(f"missing goal suite: {s} (cannot verify it held — abstaining)")

    # Per-goal floors (soft): reached the target or not, with nothing broken.
    for g in goals:
        m = by_suite.get(g.suite)
        if m is not None and m.delta < g.min_delta:
            reasons.append(f"goal {g.suite} below floor: {m.delta:.4f} < {g.min_delta:.4f}")

    # Pareto / no-trade-off (hard): no goal and no protected suite may regress beyond tol.
    goal_suites = {g.suite for g in goals}
    regressors = [m for m in candidate.metrics if (m.suite in goal_suites or m.protected) and m.delta < -max_regression]
    for m in regressors:
        reasons.append(f"regression on {m.suite}: {m.delta:.4f}")

    ret_reasons, forgetting = _retention_reasons(retention, max_regression_pct=max_retention_regression_pct, required=require_retention)
    reasons.extend(ret_reasons)

    if candidate.contaminated or regressors or forgetting:
        verdict = "reject"
    elif reasons:
        verdict = "quarantine"
    else:
        verdict = "promote"

    return PromotionDecision(
        candidate_id=candidate.id,
        verdict=verdict,
        reasons=tuple(reasons or ["all goals cleared"]),
        metrics={
            "mode": "multigoal",
            "goals": [
                {
                    "suite": g.suite,
                    "minDelta": g.min_delta,
                    "delta": by_suite[g.suite].delta if g.suite in by_suite else None,
                    "clearedFloor": g.suite in by_suite and by_suite[g.suite].delta >= g.min_delta,
                }
                for g in goals
            ],
            "maxRegression": max_regression,
            "worstRegression": min([m.delta for m in candidate.metrics] or [0.0]),
            "paretoViolations": [m.suite for m in regressors],
            "artifactCount": len(candidate.verifier_artifacts),
            "retention": _retention_metrics(retention, max_regression_pct=max_retention_regression_pct, required=require_retention, forgetting=forgetting),
            "metricRows": [m.__dict__ | {"delta": m.delta} for m in candidate.metrics],
        },
    )


def evaluate_update(candidate: UpdateCandidate, *, target_suite: str, min_target_delta: float = 0.03, max_protected_regression: float = 0.01, require_artifacts: int = 2, retention: RetentionEvidence | None = None, max_retention_regression_pct: float = 5.0, require_retention: bool = False) -> PromotionDecision:
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
    # Old-task retention: an update that improves the target suite by forgetting a
    # previously-learned task is a capability loss, not an improvement. Catastrophic
    # forgetting beyond tolerance is a HARD reject — same severity as a protected
    # regression — so the continual loop can never reward forgetting. Unverifiable
    # retention is not a silent pass: when retention is required it forces quarantine.
    forgetting = retention is not None and retention.forgot(max_retention_regression_pct)
    if forgetting:
        reasons.append(
            f"old-task retention regression: {retention.old_benchmark_delta_pct:.2f}pp "
            f"< -{abs(max_retention_regression_pct):.2f}pp (catastrophic forgetting)"
        )
    if require_retention and (retention is None or not retention.verifiable):
        reasons.append("retention evidence required but not verifiable (no learning-under-shift baseline)")
    if candidate.contaminated or regressions or forgetting:
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
            "retention": {
                "oldBenchmarkDeltaPct": retention.old_benchmark_delta_pct if retention else None,
                "passingSignal": retention.passing_signal if retention else None,
                "evaluable": retention.evaluable if retention else "not-provided",
                "maxRetentionRegressionPct": max_retention_regression_pct,
                "required": require_retention,
                "forgetting": forgetting,
            },
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


__all__ = ["EvalMetric", "RetentionEvidence", "Goal", "UpdateCandidate", "PromotionDecision", "evaluate_update", "evaluate_update_multigoal", "append_promotion_ledger", "demo_plasticity_report", "write_plasticity_report"]

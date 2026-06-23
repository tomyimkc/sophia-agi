"""Deontic hard-rule verifier for the Sophia conscience.

This is the machine-checkable subset of the constitution: prohibitions that can be
compiled to exact predicates. Optional Z3 is not required for these finite rules;
the fallback is exact and fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeonticViolation:
    rule_id: str
    severity: str
    reason: str
    counterexample: dict[str, Any]


@dataclass(frozen=True)
class DeonticDecision:
    schema: str = "sophia.deontic_verifier.v1"
    verdict: str = "accepted"  # accepted|rejected
    backend: str = "fallback"
    status: str = "consistent"
    violations: tuple[DeonticViolation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verdict": self.verdict,
            "backend": self.backend,
            "status": self.status,
            "violations": [v.__dict__ for v in self.violations],
        }


def check_deontic(action: str, *, context: dict[str, Any] | None = None) -> DeonticDecision:
    """Check hard prohibitions over action + context.

    Supported action families include publish_claim, claim_agi, write_memory,
    edit_reward, edit_constitution, train_on_hidden_eval, promote_candidate,
    weaken_verifier, and execute_tool.
    """
    context = context or {}
    a = (action or "").strip().lower()
    violations: list[DeonticViolation] = []

    def vio(rule_id: str, reason: str, severity: str = "critical") -> None:
        violations.append(DeonticViolation(rule_id, severity, reason, {"action": action, **context}))

    if a in {"claim_agi", "publish_agi_claim"} and context.get("canClaimAGI") is not True:
        vio("Forbidden(ClaimAGIWhenUnproven)", "cannot claim proven AGI while canClaimAGI != true")
    if a in {"publish_claim", "surface_claim"} and context.get("factVerdict") not in {"accepted", "non_factual"}:
        vio("Forbidden(PublishUnverifiedClaim)", "only accepted/non-factual claims may be surfaced")
    if a in {"write_memory", "write_semantic_memory", "write_procedural_memory"}:
        if context.get("memoryLayer") in {"semantic", "procedural"} and (context.get("factVerdict") != "accepted" or int(context.get("evidenceCount", 0)) <= 0):
            vio("Forbidden(TrustedMemoryWithoutEvidence)", "trusted memory requires accepted verdict and evidence")
    if a in {"edit_reward", "edit_constitution", "weaken_verifier", "lower_threshold"} and context.get("externalApproval") is not True:
        vio("Forbidden(SelfModifyRewardOrConstitution)", "reward/constitution/verifier changes require external approval")
    if a in {"train_on_hidden_eval", "reveal_hidden_eval"}:
        vio("Forbidden(HiddenEvalLeakage)", "hidden eval packs cannot be training/revealed material")
    if a == "promote_candidate" and context.get("independentRecheck") is not True:
        vio("Forbidden(SelfPromotionWithoutRecheck)", "learning candidates need independent recheck before promotion", severity="high")
    if a == "execute_tool" and context.get("moralStatus") == "PROHIBITED":
        vio("Forbidden(ExecuteProhibitedAction)", "prohibited actions cannot execute")

    return DeonticDecision(verdict="rejected" if violations else "accepted", status="violation" if violations else "consistent", violations=tuple(violations))


__all__ = ["DeonticViolation", "DeonticDecision", "check_deontic"]

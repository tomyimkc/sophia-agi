# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Mandatory Conscience enforcement adapters.

The Conscience Kernel is useful only if high-impact actions cannot bypass it.
This module turns ``agent.conscience.conscience_check`` into a small enforcement
API and a PreToolUse hook handler.

Boundary: candidate infrastructure; not AGI proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.conscience import conscience_check
from agent.hooks import HookContext, HookDecision

BLOCKING_VERDICTS = frozenset({"block", "abstain", "escalate"})
CAUTION_VERDICTS = frozenset({"retrieve", "clarify", "revise"})
HIGH_IMPACT_ACTIONS = frozenset({
    "publish_claim", "surface_claim", "finalize_answer", "execute_tool",
    "write_memory", "write_semantic_memory", "write_procedural_memory",
    "promote_candidate", "promote_verifier", "train_or_update_adapter",
    "claim_benchmark_result", "claim_agi", "publish_agi_claim",
})


@dataclass(frozen=True)
class EnforcementDecision:
    schema: str = "sophia.conscience_enforcement.v1"
    allowed: bool = True
    action: str = "draft_output"
    verdict: str = "allow"
    reason: str = "allowed by conscience"
    conscience: dict[str, Any] | None = None
    candidateOnly: bool = True
    level3Evidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "allowed": self.allowed,
            "action": self.action,
            "verdict": self.verdict,
            "reason": self.reason,
            "conscience": self.conscience,
            "candidateOnly": self.candidateOnly,
            "level3Evidence": self.level3Evidence,
        }


def enforce_conscience(
    *,
    action: str,
    text: str,
    context: dict[str, Any] | None = None,
    high_impact: bool | None = None,
    mode: str = "output",
    **kwargs: Any,
) -> EnforcementDecision:
    """Allow only if the conscience verdict is safe for the action.

    High-impact actions fail closed on ``block|abstain|escalate``. They also hold
    on ``retrieve|clarify|revise`` unless ``context['allowCautionVerdicts']`` is
    explicitly true. Low-impact diagnostics are allowed on caution verdicts but
    still block on hard blocks.
    """
    ctx = dict(context or {})
    high = (action in HIGH_IMPACT_ACTIONS) if high_impact is None else bool(high_impact)
    dec = conscience_check(text, mode=mode, action=action, context=ctx, **kwargs).to_dict()
    verdict = dec.get("verdict", "block")
    if verdict in BLOCKING_VERDICTS:
        return EnforcementDecision(allowed=False, action=action, verdict=verdict, reason=f"conscience {verdict}: {dec.get('reason', '')}", conscience=dec)
    if high and verdict in CAUTION_VERDICTS and not ctx.get("allowCautionVerdicts"):
        return EnforcementDecision(allowed=False, action=action, verdict=verdict, reason=f"high-impact action held for {verdict}: {dec.get('reason', '')}", conscience=dec)
    return EnforcementDecision(allowed=True, action=action, verdict=verdict, reason=dec.get("reason", "allowed by conscience"), conscience=dec)


def make_conscience_pretool_guard(*, default_high_impact: bool = True):
    """Return a HookBus PreToolUse handler that enforces conscience_check."""
    def _guard(ctx: HookContext) -> HookDecision | None:
        payload = ctx.payload or {}
        text = str(payload.get("text") or payload.get("content") or payload.get("claim") or ctx.args.get("text") or ctx.args.get("content") or ctx.args.get("body") or "")
        action = str(payload.get("action") or ctx.tool_id or "execute_tool")
        if not text.strip():
            return None
        result = enforce_conscience(
            action=action,
            text=text,
            context=dict(payload.get("context") or {}),
            high_impact=bool(payload.get("high_impact", default_high_impact)),
            mode="tool",
        )
        if not result.allowed:
            return HookDecision(False, result.reason, "conscience_pretool_guard")
        return None
    return _guard


__all__ = ["BLOCKING_VERDICTS", "CAUTION_VERDICTS", "HIGH_IMPACT_ACTIONS", "EnforcementDecision", "enforce_conscience", "make_conscience_pretool_guard"]

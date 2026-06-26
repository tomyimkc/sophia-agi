# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""RLVR ↔ verified-trace bridge — emit one fact+logic-stamped trace per
(case, completion) reward evaluation.

This is the wiring that turns the verified_trace.v1 logger from "fires only on
compiles / conscience checks" into "fires on every reward signal in the RLVR
training loop." Each trace carries the ACTUAL ``rl_reward.reward_for_case`` value
as its ``reward`` — from the verifier/gate seam, never a model self-score (the
selfextend rule) — so the trace log becomes a per-step provenance record of the
learning signal a GRPO run would optimize.

Stamp mapping (reuses existing verdict vocabulary, no new gates):

  fact.verdict   = "allow" if the gate passed (no forbidden attribution),
                   else "block" (a forbidden assertion is always the worst outcome)
  logic.emittable = reward >= 0  (a non-negative reward can serve as a training
                   signal; a forbidden-assertion reward of -1 is NOT emittable)
  reward         = the bounded [-1, 1] reward from rl_reward
  rewardProvenance = "rl_reward.reward_for_case" (the seam, for audit)

So a step is ``verified`` iff the gate passed AND the reward is non-negative —
exactly the completions a sound GRPO run should reinforce. A forbidden-assertion
completion lands as unverified with reward -1 and a recorded contradiction, the
signal a trainer needs to see (and the signal ``reward_is_hackable`` cross-checks
on a held-out verifier).

Offline / CI-safe: this module wraps the existing deterministic reward path and
emits traces into a (redirectable) log. It performs no training and needs no GPU;
a live GRPO run simply calls ``rewarded`` instead of ``reward_for_case`` directly
so its reward evaluations are traced.
"""
from __future__ import annotations

from typing import Any

from agent.verified_trace import VerifiedTrace, _trace_id


def rewarded(case: Any, completion: str, *, reward: float, detail: dict,
             run_id: str = "rlvr", step_idx: int = 0) -> dict:
    """Record one verified trace for an RLVR reward evaluation and return the
    trace ack (``{traceId, verified}``).

    ``reward`` and ``detail`` are the ``(reward, detail)`` returned by
    ``provenance_bench.rl_reward.reward_for_case`` (or any verifier/gate reward
    following the same contract). The fact stamp is derived from
    ``detail["passed"]``; the logic stamp from the reward sign. This is an
    observer: it never changes the reward or blocks the caller.
    """
    passed = bool(detail.get("passed"))
    # reward sign is the logic gate: a forbidden assertion scores REWARD_MIN (-1)
    # and is not emittable as a (non-negative) training signal.
    emittable = reward >= 0.0
    contradictions: list = []
    if not passed:
        # a forbidden-attribution assertion is the contradiction the gate exists to catch
        contradictions = [{"assertedForbidden": detail.get("assertedForbidden", True),
                           "reward": reward}]

    trace = VerifiedTrace(
        traceId=_trace_id(f"rlvr:{run_id}:{step_idx}:{id(case)}:{completion[:64]}"),
        runId=run_id,
        phase="rlvr",
        stepIdx=step_idx,
        claimText=str(completion)[:512],
        claimKind="derived",
        fact={
            "verdict": "allow" if passed else "block",
            "source": "provenance_faithful",
            "authorConfidence": "attributed" if detail.get("affirmsGold") else "compiled",
            "effectiveConfidenceRank": 2 if passed else 0,
            "sources": [],
        },
        logic={
            "emittable": emittable,
            "contradictions": contradictions,
            "laundered": [],
            "semanticsPreserved": True,
        },
        reward=max(-1.0, min(1.0, float(reward))),
        rewardProvenance="rl_reward.reward_for_case",
    )
    return _record_safely(trace)


def _record_safely(trace: VerifiedTrace) -> dict:
    """Record without swallowing (so callers/tests can observe the ack); the
    ``emit`` helper's swallow is for the observer-only hook sites."""
    from agent.verified_trace import record
    try:
        return record(trace)
    except Exception:  # pragma: no cover - observer-only
        return {"traceId": trace.traceId, "verified": trace.verified, "error": "record failed"}


def reward_summary(traces: list[dict]) -> dict:
    """Aggregate a slice of RLVR traces into reward + verification metrics.

    These are the metrics a live GRPO run's trace log would surface: mean reward
    over emittable completions, the share of completions that are verified
    (gate-passed AND non-negative reward), and the forbidden-assertion rate (the
    harm the gate exists to prevent). Used by the RLVR offline report.
    """
    n = len(traces) or 1
    rewards = [float(t.get("reward", 0.0)) for t in traces]
    n_verified = sum(1 for t in traces if t.get("verified"))
    n_emittable = sum(1 for t in traces if t.get("logic", {}).get("emittable"))
    n_forbidden = sum(1 for t in traces if t.get("fact", {}).get("verdict") == "block")
    return {
        "n": len(traces),
        "meanReward": round(sum(rewards) / n, 4) if traces else None,
        "meanRewardEmittable": (
            round(sum(r for r, t in zip(rewards, traces)
                      if t.get("logic", {}).get("emittable"))
                  / max(1, n_emittable), 4)
            if traces else None
        ),
        "verifiedRate": round(n_verified / n, 4) if traces else None,
        "forbiddenAssertionRate": round(n_forbidden / n, 4) if traces else None,
    }


__all__ = ["rewarded", "reward_summary"]

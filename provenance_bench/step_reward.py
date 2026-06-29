# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Process-level (per-step) RLVR reward over a verified derivation.

The step-level analogue of ``provenance_bench.math_reward`` /
``provenance_bench.physics_reward``, which reward only a *final* answer. Here the
reward is shaped by ``agent.step_verifier``: a derivation earns reward for every
machine-verified step and is penalised for any provable misstep, so a model is
trained to produce derivations whose *every step* is checkable — the
"every calculation step must be verified" objective.

Reward (bounded in [-1, 1], the RLVR contract):

    per-check value: accepted -> +1, rejected -> -1, abstain -> 0
    raw  = mean(per-check values) over all transitions (+ the gold check)
    HARD FLOOR: if ANY step is rejected (a real misstep), the reward is clamped
                to <= -1 + (fraction verified) so a single misstep cannot be
                out-voted by many trivial correct steps — a wrong derivation is
                never rewarded as if correct.

Deterministic, no model call, no GPU (sympy fails closed to abstain -> 0, never a
silent reward). Designed as the reward seam for ``tools/run_step_rlvr.py``.
"""

from __future__ import annotations

from typing import Any

from agent.step_verifier import Domain, verify_derivation


def reward_for_derivation(
    steps: "list[Any]", *, gold: str | None = None, domain: Domain = "math",
) -> tuple[float, dict[str, Any]]:
    """Return ``(score in [-1, 1], detail)`` for a proposed derivation."""
    res = verify_derivation(steps, gold=gold, default_domain=domain)
    checks = res.steps + ([res.final_check] if res.final_check else [])
    n = len(checks)
    if n == 0:
        # Nothing verifiable -> no signal (neither reward nor punish a non-derivation).
        return 0.0, {"verdict": res.verdict, "vsc": res.vsc, "nChecks": 0, "reason": "no_checks"}

    values = [1.0 if c.verdict == "accepted" else (-1.0 if c.verdict == "rejected" else 0.0) for c in checks]
    raw = sum(values) / n
    n_rejected = sum(1 for c in checks if c.verdict == "rejected")
    if n_rejected:
        # A misstep is present: cap so a wrong derivation cannot read as correct.
        frac_verified = sum(1 for c in checks if c.verdict == "accepted") / n
        raw = min(raw, -1.0 + frac_verified)
    score = max(-1.0, min(1.0, raw))
    return score, {
        "verdict": res.verdict,
        "vsc": res.vsc,
        "nChecks": n,
        "nAccepted": res.n_accepted,
        "nRejected": n_rejected,
    }

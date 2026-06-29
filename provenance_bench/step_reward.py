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

from typing import Any, Callable

from agent.step_verifier import Domain, verify_derivation

REWARD_MIN, REWARD_MAX = -1.0, 1.0

# Prepended to each problem during GRPO rollouts / eval so an instruct model emits
# a machine-parseable derivation (agent.derivation_parser reads STEP: lines first).
STEP_INSTRUCTION = (
    "Solve with an explicit, verifiable derivation. Output ONE line per step:\n"
    "  STEP: <expression> | <short justification>\n"
    "Each step's expression must equal the previous step's. End with the final "
    "answer as the last STEP.\n\n"
)


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


# --------------------------------------------------------------------------- #
# TRL GRPOTrainer wiring (mirrors provenance_bench.math_reward.make_grpo_reward).
# The reward parses the model's full completion into a derivation, then scores
# every step — so process reward, not just final-answer reward.
# --------------------------------------------------------------------------- #
def _as_list(value: Any, n: int) -> list:
    return list(value) if isinstance(value, (list, tuple)) else [value] * n


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return " ".join(
            m.get("content", "") for m in completion
            if isinstance(m, dict) and m.get("role") == "assistant"
        ) or " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def reward_for_completion(text: str, gold: str | None, *, domain: Domain = "math") -> "tuple[float, dict]":
    """Parse a model completion into steps and score the whole derivation."""
    from agent.derivation_parser import parse_derivation

    steps = parse_derivation(text or "", domain=domain)
    return reward_for_derivation(steps, gold=gold, domain=domain)


def make_grpo_reward(*, domain: Domain = "math") -> Callable:
    """TRL ``GRPOTrainer``-compatible process reward.

    Signature: ``reward_fn(prompts, completions, *, gold=None, **kwargs) -> list[float]``.
    The dataset must carry a ``gold`` column (the final-answer gold per problem).
    """
    def reward_fn(prompts: list, completions: list, *, gold: Any = None, **kwargs: Any) -> list[float]:
        n = len(completions)
        golds = _as_list(gold, n)
        out: list[float] = []
        for i, comp in enumerate(completions):
            g = golds[i] if i < len(golds) else None
            score, _ = reward_for_completion(_completion_text(comp), g, domain=domain)
            out.append(score)
        return out

    reward_fn.__name__ = f"sophia_step_process_reward_{domain}"
    return reward_fn


def offline_invariants(*, domain: Domain = "physics") -> "tuple[bool, dict]":
    """Assert the process-reward invariants (no torch, no GPU).

    Mirrors ``provenance_bench.math_reward.offline_invariants``: deterministic,
    bounded, a clean derivation scores +1, a misstep scores below it and below 0,
    the step-verifier seam is invoked, and an unverifiable completion scores 0.
    Physics (default) is sympy-independent so the checks always run in CI.
    """
    if domain == "physics":
        gold = "5 W"
        good = "STEP: 5 W | start\nSTEP: 5 J/s | watt is joule per second"
        bad = "STEP: 5 W | start\nSTEP: 5 J | WRONG unit"
    else:
        gold = "x**2 + 2*x + 1"
        good = "STEP: (x+1)**2 | start\nSTEP: x**2 + 2*x + 1 | expand"
        bad = "STEP: (x+1)**2 | start\nSTEP: x**2 - 2*x + 1 | WRONG sign"

    r_good, d_good = reward_for_completion(good, gold, domain=domain)
    r_bad, d_bad = reward_for_completion(bad, gold, domain=domain)
    r_repeat, _ = reward_for_completion(good, gold, domain=domain)
    r_empty, _ = reward_for_completion("no idea", None, domain=domain)

    checks = {
        "deterministic": r_good == r_repeat,
        "bounded": all(REWARD_MIN <= r <= REWARD_MAX for r in (r_good, r_bad, r_empty)),
        "cleanPositive": r_good == REWARD_MAX,
        "misstepBelowClean": r_bad < r_good,
        "misstepNegative": r_bad < 0.0,
        "seamInvoked": d_bad.get("verdict") == "rejected",
        "unverifiableZero": r_empty == 0.0,
    }
    detail = {"domain": domain, "rewards": {"good": d_good, "bad": d_bad}, "checks": checks}
    return all(checks.values()), detail


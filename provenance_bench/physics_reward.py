# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-as-reward for PHYSICS — dimensional + numeric equivalence is the reward.

The math RLVR reward (``math_reward``) uses symbolic equivalence as the seam; this
is the physics analogue: the reward IS whether the model's final answer is the same
*physical quantity* as the gold — identical SI dimension AND value within 1% — via
``agent.verifiers.physics_equivalent``. Dimensional analysis is the ground truth
(``9.8 J`` ≠ ``9.8 m/s^2``); it is pure-Python, judge-free, and ungameable by the
right-number-wrong-unit failure mode.

Reward in ``[-1, 1]``: +1 equivalent, -1 otherwise. The units engine is stdlib-only,
so (unlike sympy for math) there is no "backend unavailable" abstention for numeric
golds — only a *symbolic* gold can fall through to sympy and, if it is absent, score
-1 with ``sympy_unavailable`` (a held verdict, never a silent +1).

TRL wiring mirrors ``math_reward.make_grpo_reward``: the dataset (``physics_dataset``)
carries a ``gold`` column so it arrives via ``**kwargs`` aligned with completions.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.verifiers import physics_equivalent

REWARD_MIN, REWARD_MAX = -1.0, 1.0


def reward_for_problem(
    answer: str,
    gold: str,
    *,
    rtol: float = 1e-2,
    extract: bool = True,
    spy: dict | None = None,
) -> "tuple[float, dict]":
    """Deterministic physics reward: +1 iff ``answer`` matches ``gold`` (dimension
    AND value within ``rtol``).

    ``spy`` is an optional mutable dict incremented on each verifier call, so a test
    can prove the physics_equivalent seam was actually invoked.
    """
    res = physics_equivalent(gold, rtol=rtol, extract=extract)(answer or "", None, {})
    if spy is not None:
        spy["verifier_calls"] = spy.get("verifier_calls", 0) + 1
    passed = bool(res["passed"])
    score = REWARD_MAX if passed else REWARD_MIN
    detail = {
        "passed": passed,
        "reason": (res.get("reasons") or [""])[0] if not passed else "equivalent",
        "gold": gold,
        "got": (res.get("detail") or {}).get("got"),
        "relErr": (res.get("detail") or {}).get("relErr"),
        "reward": score,
    }
    return (score, detail)


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


def make_grpo_reward(*, rtol: float = 1e-2, extract: bool = True) -> Callable:
    """TRL ``GRPOTrainer``-compatible physics reward.

    Signature: ``reward_fn(prompts, completions, *, gold=None, **kwargs) -> list[float]``.
    The dataset must carry a ``gold`` column (the physical-quantity gold per problem)
    so it arrives here via ``**kwargs`` aligned with completions.
    """
    def reward_fn(prompts: list, completions: list, *, gold: Any = None, **kwargs: Any) -> list[float]:
        n = len(completions)
        golds = _as_list(gold, n)
        out: list[float] = []
        for i, comp in enumerate(completions):
            g = golds[i] if i < len(golds) else ""
            score, _ = reward_for_problem(_completion_text(comp), g or "", rtol=rtol, extract=extract)
            out.append(score)
        return out

    reward_fn.__name__ = "sophia_physics_equivalent_reward"
    return reward_fn


def offline_invariants() -> "tuple[bool, dict]":
    """Assert the physics reward-machinery invariants (no torch, no GPU).

    Mirrors ``math_reward.offline_invariants``: deterministic, monotone in the right
    direction, wrong-answer negative, the physics_equivalent seam is actually
    invoked, the reward is bounded, the family split is contamination-free, AND — the
    physics-specific check — a right-number/wrong-unit answer is negative (the failure
    mode dimensional analysis exists to catch). No optional backend required.
    """
    from provenance_bench import physics_dataset

    spy = {"verifier_calls": 0}

    gold = "30 N"
    good = r"By F = m a, the answer is \boxed{30 N}."
    bad_value = r"The answer is \boxed{31.5 N}."          # right dim, value off by 5%
    bad_unit = r"The answer is \boxed{30 J}."             # right number, wrong dimension

    r_good, d_good = reward_for_problem(good, gold, spy=spy)
    r_bad_v, d_bad_v = reward_for_problem(bad_value, gold, spy=spy)
    r_bad_u, d_bad_u = reward_for_problem(bad_unit, gold, spy=spy)
    r_repeat, _ = reward_for_problem(good, gold, spy=spy)

    data = physics_dataset.build_physics_rl_dataset(eval_frac=0.4, seed=0)

    checks = {
        "deterministic": r_good == r_repeat,
        "correctPositive": r_good == REWARD_MAX,
        "wrongValueNegative": r_bad_v == REWARD_MIN,
        "wrongUnitNegative": r_bad_u == REWARD_MIN,
        "monotone": r_good > r_bad_v and r_good > r_bad_u,
        "verifierSeamInvoked": spy["verifier_calls"] >= 4,
        "bounded": all(REWARD_MIN <= r <= REWARD_MAX for r in (r_good, r_bad_v, r_bad_u)),
        "contaminationFree": len(data["family_intersection"]) == 0,
        "trainNonEmpty": len(data["train_rows"]) > 0,
        "evalNonEmpty": len(data["eval_rows"]) > 0,
    }
    detail = {
        "rewards": {"good": d_good, "badValue": d_bad_v, "badUnit": d_bad_u},
        "checks": checks,
        "trainProblems": len(data["train_problems"]),
        "evalProblems": len(data["eval_problems"]),
        "trainFamilies": sorted({p["family"] for p in data["train_problems"]}),
        "evalFamilies": sorted({p["family"] for p in data["eval_problems"]}),
        "trainSealed": data["train_sealed"],
        "evalSealed": data["eval_sealed"],
        "familyIntersection": data["family_intersection"],
    }
    return all(v for v in checks.values()), detail

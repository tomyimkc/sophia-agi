# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-as-reward for MATH — symbolic equivalence is the reward.

The code RLVR reward (``code_reward``) uses tests-pass as the seam; this is the
math analogue: the reward IS whether the model's final answer is *symbolically
equivalent* to the gold answer (``agent.verifiers.math_equivalent`` → sympy). Like
the interpreter for code, the CAS is ground truth — no LLM judge, ungameable.

Reward in ``[-1, 1]``: +1 equivalent, -1 otherwise. **Fail-closed:** when sympy is
absent ``math_equivalent`` cannot verify, so the reward is -1 with reason
``sympy_unavailable`` (a held verdict, never a silent +1). Install
``requirements-math.txt`` to enable the algebra.

TRL wiring mirrors ``code_reward.make_grpo_reward``: the dataset (``math_dataset``)
carries a ``gold`` column so it arrives via ``**kwargs`` aligned with completions.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.verifiers import math_equivalent

REWARD_MIN, REWARD_MAX = -1.0, 1.0


def reward_for_problem(
    answer: str,
    gold: str,
    *,
    extract: bool = True,
    spy: dict | None = None,
) -> "tuple[float, dict]":
    """Deterministic math reward: +1 iff ``answer`` ≡ ``gold`` symbolically.

    ``spy`` is an optional mutable dict incremented on each verifier call, so a
    test can prove the math_equivalent seam was actually invoked.
    """
    res = math_equivalent(gold, extract=extract)(answer or "", None, {})
    if spy is not None:
        spy["verifier_calls"] = spy.get("verifier_calls", 0) + 1
    passed = bool(res["passed"])
    score = REWARD_MAX if passed else REWARD_MIN
    detail = {
        "passed": passed,
        "reason": (res.get("reasons") or [""])[0] if not passed else "equivalent",
        "sympy": bool((res.get("detail") or {}).get("sympy", False)),
        "gold": gold,
        "got": (res.get("detail") or {}).get("got"),
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


def make_grpo_reward(*, extract: bool = True) -> Callable:
    """TRL ``GRPOTrainer``-compatible math reward.

    Signature: ``reward_fn(prompts, completions, *, gold=None, **kwargs) -> list[float]``.
    The dataset must carry a ``gold`` column (the symbolic gold answer per problem)
    so it arrives here via ``**kwargs`` aligned with completions.
    """
    def reward_fn(prompts: list, completions: list, *, gold: Any = None, **kwargs: Any) -> list[float]:
        n = len(completions)
        golds = _as_list(gold, n)
        out: list[float] = []
        for i, comp in enumerate(completions):
            g = golds[i] if i < len(golds) else ""
            score, _ = reward_for_problem(_completion_text(comp), g or "", extract=extract)
            out.append(score)
        return out

    reward_fn.__name__ = "sophia_math_equivalent_reward"
    return reward_fn


def sympy_available() -> bool:
    try:
        import sympy  # noqa: F401
        return True
    except Exception:
        return False


def offline_invariants() -> "tuple[bool, dict]":
    """Assert the math reward-machinery invariants (no torch, no GPU).

    Mirrors ``tools/run_rlvr._offline_invariants`` for the math task: deterministic,
    monotone in the right direction, wrong-answer negative, the math_equivalent seam
    is actually invoked, the reward is bounded, and the family split is
    contamination-free. The algebra checks require sympy; when it is absent they are
    reported skipped (the structural checks still run) rather than failing CI.
    """
    from provenance_bench import math_dataset

    have_sympy = sympy_available()
    spy = {"verifier_calls": 0}

    # A correct and a wrong answer to the same problem (factoring).
    gold = "(x-1)*(x+1)"
    good = r"After factoring, the answer is \boxed{(x - 1)(x + 1)}."
    bad = r"The answer is \boxed{x**2 + 1}."

    r_good, d_good = reward_for_problem(good, gold, spy=spy)
    r_bad, d_bad = reward_for_problem(bad, gold, spy=spy)
    r_repeat, _ = reward_for_problem(good, gold, spy=spy)

    data = math_dataset.build_math_rl_dataset(eval_frac=0.34, seed=0)

    checks = {
        "deterministic": r_good == r_repeat,
        "verifierSeamInvoked": spy["verifier_calls"] >= 3,
        "bounded": all(REWARD_MIN <= r <= REWARD_MAX for r in (r_good, r_bad)),
        "contaminationFree": len(data["family_intersection"]) == 0,
        "trainNonEmpty": len(data["train_rows"]) > 0,
        "evalNonEmpty": len(data["eval_rows"]) > 0,
    }
    # Algebra-dependent invariants (only meaningful with sympy).
    if have_sympy:
        checks["correctPositive"] = r_good == REWARD_MAX
        checks["wrongNegative"] = r_bad == REWARD_MIN
        checks["monotone"] = r_good > r_bad
    else:
        checks["sympySkipped"] = True

    detail = {
        "sympy": have_sympy,
        "rewards": {"good": d_good, "bad": d_bad},
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

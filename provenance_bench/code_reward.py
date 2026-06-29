# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-as-reward for CODE — tests-pass is the reward (the strongest RLVR signal).

The provenance RLVR reward (``rl_reward``) uses ``provenance_faithful`` as the
seam. This is the code analogue: the reward IS whether the model's solution passes
the hidden canonical tests when executed (``provenance_bench.code_exec``). Unlike a
humanities judge, this signal is *objective* — the interpreter decides. It is NOT,
however, ungameable: the solution runs in the same process as the appended test, so
it can manipulate the exit code (``sys.exit(0)``/``atexit``), override ``__eq__``,
tamper with the harness, or special-case the visible inputs and still score +1
(see the ``code-reward-hackable-not-ungameable`` failure-ledger entry; the demo is
in ``tests/test_code_integrity.py``). Use ``provenance_bench.code_integrity`` (the
integrity-gated composite reward) for any run that trains on this signal — it floors
detected reward-hacks before the executor is consulted. This is the RLVR setup that
works best (DeepSeek-R1 code RL) *once the verifier itself is hardened*.

Reward in ``[-1, 1]``: +1 tests pass, -1 code present but fails, -1 no code at all
(the model must actually answer with code). Deterministic given the sandbox.

TRL wiring mirrors ``rl_reward.make_grpo_reward``: the dataset carries a ``test``
column (the hidden test) so it arrives via ``**kwargs``.
"""

from __future__ import annotations

from typing import Any, Callable

from provenance_bench.code_exec import check_answer

REWARD_MIN, REWARD_MAX = -1.0, 1.0


def reward_for_task(
    answer: str,
    test_code: str,
    *,
    timeout_sec: int = 15,
    spy: dict | None = None,
) -> "tuple[float, dict]":
    """Deterministic code reward: +1 iff the answer's code passes ``test_code``.

    ``spy`` is an optional mutable dict incremented on each verifier call, so a
    test can prove the tests-pass seam was actually invoked.
    """
    res = check_answer(answer, test_code, timeout_sec=timeout_sec)
    if spy is not None:
        spy["verifier_calls"] = spy.get("verifier_calls", 0) + 1
    score = REWARD_MAX if res["passed"] else REWARD_MIN
    return (score, {"passed": res["passed"], "reason": res["reason"], "executed": res.get("executed")})


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


def make_grpo_reward(*, timeout_sec: int = 15) -> Callable:
    """TRL ``GRPOTrainer``-compatible code reward.

    Signature: ``reward_fn(prompts, completions, *, test=None, **kwargs) -> list[float]``.
    The dataset must carry a ``test`` column (the hidden canonical test per task) so
    it arrives here via ``**kwargs`` aligned with completions.
    """
    def reward_fn(prompts: list, completions: list, *, test: Any = None, **kwargs: Any) -> list[float]:
        n = len(completions)
        tests = _as_list(test, n)
        out: list[float] = []
        for i, comp in enumerate(completions):
            t = tests[i] if i < len(tests) else ""
            score, _ = reward_for_task(_completion_text(comp), t or "", timeout_sec=timeout_sec)
            out.append(score)
        return out

    reward_fn.__name__ = "sophia_code_tests_reward"
    return reward_fn


def exec_enabled() -> bool:
    """Whether ``code_exec`` will actually execute (vs syntax-only fallback)."""
    from provenance_bench.code_exec import _exec_on

    return _exec_on()


def offline_invariants() -> "tuple[bool, dict]":
    """Assert the code reward-machinery invariants (no torch, no GPU).

    Mirrors ``math_reward.offline_invariants`` for the code task: deterministic,
    monotone in the right direction, a wrong solution scores negative, the
    tests-pass seam is actually invoked, the reward is bounded, and the family
    split is contamination-free. The correctness checks require the executor to
    actually run (``SOPHIA_ALLOW_CODE_EXEC=1``); when execution is off they are
    reported skipped (the structural checks still run) rather than failing CI,
    because syntax-only cannot decide correctness.
    """
    from provenance_bench import code_dataset

    can_exec = exec_enabled()
    spy = {"verifier_calls": 0}

    # A correct and a wrong solution to the SAME hidden test.
    test_code = "assert scale(3, 4) == 12\nassert scale(0, 5) == 0\n"
    good = "```python\ndef scale(n, k):\n    return n * k\n```"
    bad = "```python\ndef scale(n, k):\n    return n + k\n```"

    r_good, d_good = reward_for_task(good, test_code, spy=spy)
    r_bad, d_bad = reward_for_task(bad, test_code, spy=spy)
    r_repeat, _ = reward_for_task(good, test_code, spy=spy)

    data = code_dataset.build_code_rl_dataset(eval_frac=0.34, seed=0)

    checks = {
        "deterministic": r_good == r_repeat,
        "codeSeamInvoked": spy["verifier_calls"] >= 3,
        "bounded": all(REWARD_MIN <= r <= REWARD_MAX for r in (r_good, r_bad)),
        "contaminationFree": len(data["family_intersection"]) == 0,
        "trainNonEmpty": len(data["train_rows"]) > 0,
        "evalNonEmpty": len(data["eval_rows"]) > 0,
    }
    # Execution-dependent invariants (only meaningful when the interpreter runs).
    if can_exec:
        checks["correctPositive"] = r_good == REWARD_MAX
        checks["wrongNegative"] = r_bad == REWARD_MIN
        checks["monotone"] = r_good > r_bad
    else:
        checks["execSkipped"] = True

    detail = {
        "exec": can_exec,
        "rewards": {"good": d_good, "bad": d_bad},
        "checks": checks,
        "trainTasks": len(data["train_tasks"]),
        "evalTasks": len(data["eval_tasks"]),
        "trainFamilies": sorted({t["family"] for t in data["train_tasks"]}),
        "evalFamilies": sorted({t["family"] for t in data["eval_tasks"]}),
        "trainSealed": data["train_sealed"],
        "evalSealed": data["eval_sealed"],
        "familyIntersection": data["family_intersection"],
    }
    return all(v for v in checks.values()), detail

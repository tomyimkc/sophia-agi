"""Verifier-as-reward for CODE — tests-pass is the reward (the strongest RLVR signal).

The provenance RLVR reward (``rl_reward``) uses ``provenance_faithful`` as the
seam. This is the code analogue: the reward IS whether the model's solution passes
the hidden canonical tests when executed (``provenance_bench.code_exec``). Unlike a
humanities judge, this signal is objective and ungameable — the interpreter
decides. This is exactly the RLVR setup that works best (DeepSeek-R1 code RL).

Reward in ``[-1, 1]``: +1 tests pass, -1 code present but fails, -1 no code at all
(the model must actually answer with code). Deterministic given the sandbox.

TRL wiring mirrors ``rl_reward.make_grpo_reward``: the dataset carries a ``test``
column (the hidden test) so it arrives via ``**kwargs``.
"""

from __future__ import annotations

from typing import Any, Callable

from provenance_bench.code_exec import check_answer

REWARD_MIN, REWARD_MAX = -1.0, 1.0


def reward_for_task(answer: str, test_code: str, *, timeout_sec: int = 15) -> "tuple[float, dict]":
    """Deterministic code reward: +1 iff the answer's code passes ``test_code``."""
    res = check_answer(answer, test_code, timeout_sec=timeout_sec)
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

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-as-reward for VISION — grounding is the reward (multimodal RLVR).

The math RLVR reward (``math_reward``) makes symbolic equivalence the reward; the
code reward makes tests-pass the reward. This is the multimodal analogue: the
reward IS whether the model's answer matches the **judge-free scene verifier**
(``multimodal_bench/verifiers.py``) — the CAS/interpreter of the visual world. No
LLM judge sits in the reward, so it is ungameable.

Unlike the binary math/code rewards, the visual reward is *calibration-aware*, the
distinctive Sophia design (VISION.md: fail-closed, honest abstention):

    correct (grounded)          -> +1.0
    abstained ("I can't tell")  -> -0.25   (honest, mildly costly)
    wrong (confident error)     -> -1.0   (confident hallucination is the worst)

so the ordering is ``correct > abstain > wrong`` — the model is trained to prefer
saying "I don't know" over hallucinating, exactly the behaviour the trap suite
measures. **Fail-closed:** a trap whose gold is *not* confirmed by the verifier at
reward time is held at the wrong-reward with reason ``verifier_mismatch`` (never a
silent +1). The verifier seam (``verifiers.resolve_check``) is invoked on every
call, so the reward can never drift from machine ground truth.

TRL wiring mirrors ``math_reward.make_grpo_reward``: the dataset carries a
``trap`` column (the full trap dict per row) so it arrives via ``**kwargs``.
"""

from __future__ import annotations

from typing import Any, Callable

from multimodal_bench import judge as judge_mod
from multimodal_bench import verifiers

# Default calibration-aware reward surface. Override per-call for ablations.
DEFAULT_WEIGHTS = {"correct": 1.0, "abstain": -0.25, "wrong": -1.0}
REWARD_MIN, REWARD_MAX = -1.0, 1.0


def reward_for_trap(
    answer: str, trap: dict, *, weights: "dict | None" = None, spy: "dict | None" = None,
) -> "tuple[float, dict]":
    """Deterministic visual reward derived from the judge-free verifier.

    ``spy`` (optional) is incremented on each verifier invocation so a test can
    prove the seam was actually exercised.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    # Fail-closed: only trust a trap whose human gold is confirmed by the verifier.
    confirmed = verifiers.gold_matches_check(trap)
    if spy is not None:
        spy["verifier_calls"] = spy.get("verifier_calls", 0) + 1
    if not confirmed:
        return (w["wrong"], {"category": "held", "reason": "verifier_mismatch",
                             "reward": w["wrong"], "groundTruth": None})

    gold_value = verifiers.resolve_check(trap["scene"], trap["check"])
    # The judge here is the deterministic LEXICAL parser (no LLM) — it maps the
    # free-text answer onto the verifier-confirmed gold.
    v = judge_mod.lexical_judge(answer, trap)
    if v.abstained:
        cat = "abstain"
    elif v.affirmed_gold:
        cat = "correct"
    else:
        cat = "wrong"
    score = w[cat]
    return (score, {"category": cat, "reason": cat, "reward": score,
                    "hallucinated": bool(v.hallucinated), "groundTruth": gold_value})


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


def make_grpo_reward(*, weights: "dict | None" = None) -> Callable:
    """TRL ``GRPOTrainer``-compatible visual reward.

    Signature: ``reward_fn(prompts, completions, *, trap=None, **kwargs) -> list[float]``.
    The dataset must carry a ``trap`` column (the full trap dict per row) so it
    arrives here via ``**kwargs`` aligned with completions.
    """
    def reward_fn(prompts: list, completions: list, *, trap: Any = None, **kwargs: Any) -> list[float]:
        n = len(completions)
        traps = _as_list(trap, n)
        out: list[float] = []
        for i, comp in enumerate(completions):
            t = traps[i] if i < len(traps) else None
            if not t:
                out.append(REWARD_MIN)
                continue
            score, _ = reward_for_trap(_completion_text(comp), t, weights=weights)
            out.append(score)
        return out

    reward_fn.__name__ = "sophia_visual_grounding_reward"
    return reward_fn


def offline_invariants() -> "tuple[bool, dict]":
    """Assert the visual reward-machinery invariants (no torch, no GPU).

    Mirrors ``math_reward.offline_invariants``: deterministic, bounded, the honesty
    ordering correct > abstain > wrong holds, the verifier seam is invoked, and the
    RL split is contamination-free.
    """
    from multimodal_bench import visual_dataset

    # A yes/no trap with gold 'no' (a phantom-object style row).
    trap = {
        "id": "inv-phantom", "category": "phantom_object", "answer_type": "yesno",
        "gold_answer": "no", "trap_answer": "yes",
        "scene": {"objects": [{"label": "sofa", "box": [0, 0, 10, 10]}], "texts": []},
        "check": {"type": "presence", "label": "cat", "expect": False},
    }
    spy = {"verifier_calls": 0}
    r_correct, d_correct = reward_for_trap("No, there is no cat.", trap, spy=spy)
    r_abstain, d_abstain = reward_for_trap("I can't tell from this image.", trap, spy=spy)
    r_wrong, d_wrong = reward_for_trap("Yes, a cat is on the sofa.", trap, spy=spy)
    r_repeat, _ = reward_for_trap("No, there is no cat.", trap, spy=spy)

    data = visual_dataset.build_visual_rl_dataset(eval_frac=0.34, seed=0)

    checks = {
        "deterministic": r_correct == r_repeat,
        "correctPositive": r_correct == REWARD_MAX,
        "wrongNegative": r_wrong == REWARD_MIN,
        "honestyOrdering": r_correct > r_abstain > r_wrong,
        "verifierSeamInvoked": spy["verifier_calls"] >= 4,
        "bounded": all(REWARD_MIN <= r <= REWARD_MAX for r in (r_correct, r_abstain, r_wrong)),
        "contaminationFree": len(data["family_intersection"]) == 0,
        "trainNonEmpty": len(data["train_rows"]) > 0,
        "evalNonEmpty": len(data["eval_rows"]) > 0,
    }
    detail = {
        "rewards": {"correct": d_correct, "abstain": d_abstain, "wrong": d_wrong},
        "checks": checks,
        "trainFamilies": data["train_families"],
        "evalFamilies": data["eval_families"],
        "familyIntersection": data["family_intersection"],
        "trainRows": len(data["train_rows"]),
        "evalRows": len(data["eval_rows"]),
    }
    return all(checks.values()), detail

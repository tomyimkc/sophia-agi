# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Rollout-driven GRPO for the retrieval-faithfulness reward.

The vanilla TRL ``GRPOTrainer`` reward callback only sees completion text, but the
faithfulness reward needs a full trajectory (retrieval context + the counterfactual
citation-drop regeneration). So the SAMPLING unit here is a whole
``faithfulness_rollout.rollout`` — for each case we sample a GROUP of G rollouts,
score each with ``retrieval_faithfulness.reward_for_trajectory``, and compute the
GRPO group-relative advantage. A live trainer applies those advantages as the
policy-gradient weight on each rollout's tokens (no value network — the group mean
is the baseline, à la DeepSeek-R1 GRPO).

This module ships the loop's deterministic core (group advantage, within-group
spread, sampling) and its offline invariants. The headline property it proves:

  ** When a correctness-only reward would COLLAPSE (every rollout is "correct", so
     within-group reward variance is 0 and the advantage is 0 -> no learning
     signal), the faithfulness reward still SEPARATES a retrieval-using rollout
     from a weights-leaking one -> non-zero advantage -> the policy can learn to be
     faithful. **

This mirrors the repo's existing multi-axis anti-reward-collapse thesis
(agent.multiaxis_reward / run_rlvr `--reward multiaxis`), applied to faithfulness.

The live gradient step needs torch + a policy with per-rollout token logprobs;
``run_live`` is an honest gate (Open in agi-proof/failure-ledger.md), not a trained
model. The offline invariants run with no torch / GPU / corpus.
"""

from __future__ import annotations

import statistics
from typing import Any, Callable


def within_group_std(rewards: list) -> float:
    """Population std of a group's rewards — the GRPO learning signal. Zero means
    every rollout got the same reward (advantage collapses to zero)."""
    if len(rewards) < 2:
        return 0.0
    return statistics.pstdev(rewards)


def group_advantages(rewards: list, *, eps: float = 1e-6) -> list:
    """GRPO group-relative, value-free advantage: ``(r - mean) / std``. A collapsed
    group (std <= eps) yields all-zero advantages (no gradient), which is exactly the
    failure mode the faithfulness term is designed to avoid on all-correct groups."""
    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    sd = within_group_std(rewards)
    if sd <= eps:
        return [0.0] * len(rewards)
    return [(r - mean) / sd for r in rewards]


def sample_group(
    case: dict,
    policies: list,
    *,
    seams: dict,
    reward_fn: Callable | None = None,
) -> list:
    """Roll out one GROUP for ``case`` (one rollout per policy in ``policies``) and
    return ``[{traj, reward, detail}, ...]``.

    Offline, the group's diversity comes from passing distinct deterministic
    ``policies``; live, it is ``[stochastic_policy] * G`` sampled at temperature. ``seams``
    supplies ``retrieve`` / ``extract_claims`` / ``verify_claim`` (+ optional
    ``check_correct``); ``reward_fn`` defaults to the faithfulness reward."""
    from provenance_bench.faithfulness_rollout import rollout
    from provenance_bench.retrieval_faithfulness import reward_for_trajectory

    rf = reward_fn or reward_for_trajectory
    out = []
    for policy in policies:
        traj = rollout(case, generate=policy, **seams)
        reward, detail = rf(traj)
        out.append({"traj": traj, "reward": reward, "detail": detail})
    return out


def offline_invariants() -> tuple[bool, dict]:
    """Assert the GRPO advantage math + the anti-collapse property (no torch/GPU).

    Builds a group of four all-CORRECT rollouts — two from a retrieval-using
    (faithful) policy, two from a weights-leaking one — on an identical answer, then
    shows a correctness-only reward collapses (std 0, zero advantage) while the
    faithfulness reward keeps a non-zero learning signal and assigns the faithful
    rollouts positive advantage, the leaky ones negative."""
    from provenance_bench import faithfulness_rollout as fr

    case = {"prompt": "Who wrote the Project Phoenix Charter?",
            "should_retrieve": True, "answerable": True, "gold": "founding committee"}
    seams = dict(retrieve=fr._mock_retrieve, extract_claims=fr._mock_extract,
                 verify_claim=fr._mock_verify, check_correct=fr._check_correct)
    # Group order: faithful, faithful, leaky, leaky.
    policies = [fr._faithful_policy, fr._faithful_policy, fr._leaky_policy, fr._leaky_policy]

    group = sample_group(case, policies, seams=seams)
    rewards = [g["reward"] for g in group]
    adv = group_advantages(rewards)

    # The shadow "correctness-only" reward: 1.0 for every correct rollout. All four are
    # correct, so this collapses — the exact case where faithfulness must still teach.
    correctness_only = [1.0 if g["traj"].get("task_correct") else 0.0 for g in group]

    checks = {
        "allRolloutsCorrect": all(c == 1.0 for c in correctness_only),
        "correctnessOnlyCollapses": within_group_std(correctness_only) == 0.0,
        "faithfulnessGivesSignal": within_group_std(rewards) > 0.0,
        "advantagesSumZero": abs(sum(adv)) < 1e-6,
        "faithfulPositiveLeakyNegative": adv[0] > 0.0 > adv[2],
        "collapsedGroupZeroAdvantage": group_advantages([0.5, 0.5, 0.5]) == [0.0, 0.0, 0.0],
        "advantagesFinite": all(-1e3 < a < 1e3 for a in adv),
    }
    detail = {
        "checks": checks,
        "rewards": [round(r, 4) for r in rewards],
        "advantages": [round(a, 4) for a in adv],
        "withinGroupStd": round(within_group_std(rewards), 4),
        "correctnessOnlyStd": within_group_std(correctness_only),
        "note": "correctness-only collapses to zero advantage; faithfulness still separates "
                "retrieval-using from weights-leaking rollouts (the anti-collapse property).",
    }
    return all(checks.values()), detail


def run_live(*_args: Any, **_kwargs: Any) -> int:
    """Live rollout-driven GRPO. Gated: needs torch + a policy exposing per-rollout
    token logprobs to apply ``group_advantages`` as the policy-gradient weight. The
    loop structure (sample_group -> group_advantages -> weighted update) is validated
    offline; the gradient integration is Open in the failure ledger."""
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"Install RL deps: pip install -r requirements-rl.txt ({type(exc).__name__}: {exc})")
        return 1
    print(
        "Live faithfulness GRPO is not yet wired to a policy-gradient backend: the loop "
        "(sample_group -> group_advantages -> per-rollout weighted update) is offline-"
        "validated, but applying advantages as token-level PG weights with the counterfactual "
        "regeneration in the sampling path is Open in agi-proof/failure-ledger.md. Run "
        "`python tools/run_rlvr.py --task faithfulness --model mock` for the offline invariants."
    )
    return 1


__all__ = ["within_group_std", "group_advantages", "sample_group", "offline_invariants", "run_live"]

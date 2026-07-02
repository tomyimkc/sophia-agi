# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A4 — frugal-RL curation + advantage shaping (from Agents-A1, arXiv 2606.30616).

Three compute-frugal tricks from the Agents-A1 teacher recipes, implemented as
pure, deterministic functions so their invariants are testable offline:

1. **Mixed-outcome filtering** (their §4.2.1): given N pre-attempt rewards per
   prompt, keep only prompts whose outcomes are MIXED (neither always-success
   nor always-failure). Uniform prompts carry zero GRPO advantage — training on
   them wastes rollouts.
2. **Dynamic sampling** (their §4.2.3): the same predicate applied per rollout
   group at training time (drop zero-variance groups).
3. **PAPO-style asymmetric advantage shaping** (their §4.2.4, Eq. 8):
   A_i = A_i^out + lambda_neg * 1[failed_i] * A_i^proc, where A^out is the
   group-normalized outcome reward and A^proc is the process score normalized
   over the group's FAILED members only — process reward ranks failures by
   closeness to success and never double-counts successes.

GRPO-compat note (honest approximation, do not overclaim): TRL's GRPOTrainer
re-normalizes whatever the reward function returns into per-group advantages
(affine per group). ``make_papo_group_reward`` therefore returns the PAPO
advantage AS the reward, which preserves PAPO's within-group ordering and
relative weighting up to that per-group affine transform — it is
"PAPO-shaped", not bit-identical PAPO. candidateOnly: any uplift claim still
requires the gated harness.
"""
from __future__ import annotations

import statistics
from typing import Any, Callable, Sequence

DEFAULT_LAMBDA_NEG = 0.5


def is_mixed_outcome(rewards: Sequence[float], *, success_threshold: float = 0.5) -> bool:
    """True iff the attempts contain BOTH a success and a failure."""
    if not rewards:
        return False
    outcomes = [r >= success_threshold for r in rewards]
    return any(outcomes) and not all(outcomes)


def mixed_outcome_keep(rewards_per_prompt: "Sequence[Sequence[float]]",
                       *, success_threshold: float = 0.5) -> list[bool]:
    """Keep-mask over prompts: True where pre-attempt outcomes are mixed."""
    return [is_mixed_outcome(rs, success_threshold=success_threshold)
            for rs in rewards_per_prompt]


def dynamic_group_keep(group_rewards: Sequence[float], *, eps: float = 1e-9) -> bool:
    """True iff the rollout group has non-degenerate reward variance."""
    if len(group_rewards) < 2:
        return False
    return statistics.pstdev(group_rewards) > eps


def _normalize(values: Sequence[float]) -> list[float]:
    """Group-normalize to zero mean / unit std; all-equal groups -> zeros."""
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    std = statistics.pstdev(values)
    if std <= 1e-12:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def papo_advantages(outcome_rewards: Sequence[float],
                    process_scores: Sequence[float],
                    *, lambda_neg: float = DEFAULT_LAMBDA_NEG,
                    success_threshold: float = 0.5) -> list[float]:
    """PAPO-style asymmetric advantages for ONE rollout group (Agents-A1 Eq. 8).

    A_i = A_i^out + lambda_neg * 1[r_i^out failed] * A_i^proc, with A^out
    normalized over the whole group and A^proc normalized over FAILED members
    only. Successes are never shaped by the process score (they already satisfy
    it — adding it would double-count); the process term only ranks failures by
    closeness to success.
    """
    if len(outcome_rewards) != len(process_scores):
        raise ValueError(
            f"outcome/process length mismatch: {len(outcome_rewards)} != {len(process_scores)}")
    a_out = _normalize(outcome_rewards)
    failed_idx = [i for i, r in enumerate(outcome_rewards) if r < success_threshold]
    a_proc_failed = _normalize([process_scores[i] for i in failed_idx])
    a_proc = [0.0] * len(outcome_rewards)
    for j, i in enumerate(failed_idx):
        a_proc[i] = a_proc_failed[j]
    return [a_out[i] + lambda_neg * a_proc[i] for i in range(len(outcome_rewards))]


def make_papo_group_reward(outcome_fn: Callable, process_fn: Callable,
                           *, num_generations: int,
                           lambda_neg: float = DEFAULT_LAMBDA_NEG,
                           success_threshold: float = 0.5) -> Callable:
    """Wrap outcome+process reward fns into a PAPO-shaped GRPO reward.

    ``outcome_fn``/``process_fn`` follow the TRL contract
    (fn(prompts, completions, **kw) -> list[float]). Rewards are shaped per
    contiguous group of ``num_generations`` completions (TRL's grouping).
    """
    def reward_fn(prompts: list, completions: list, **kwargs: Any) -> list[float]:
        r_out = outcome_fn(prompts, completions, **kwargs)
        r_proc = process_fn(prompts, completions, **kwargs)
        g = max(1, int(num_generations))
        shaped: list[float] = []
        for i in range(0, len(r_out), g):
            shaped.extend(papo_advantages(
                r_out[i:i + g], r_proc[i:i + g],
                lambda_neg=lambda_neg, success_threshold=success_threshold))
        return shaped

    reward_fn.__name__ = "sophia_papo_shaped_reward"
    return reward_fn


def offline_invariants(*, lambda_neg: float = DEFAULT_LAMBDA_NEG) -> "tuple[bool, dict]":
    """Assert the curation/shaping invariants (no torch, no GPU)."""
    # mixed-outcome filter
    keep = mixed_outcome_keep([[1.0, 0.0, 1.0], [1.0, 1.0], [0.0, 0.0], []])
    # PAPO on one group: two successes, two failures with different process scores
    r_out = [1.0, 1.0, 0.0, 0.0]
    r_proc = [1.0, 1.0, 0.8, 0.2]  # near-miss failure vs far failure
    adv = papo_advantages(r_out, r_proc, lambda_neg=lambda_neg)
    adv0 = papo_advantages(r_out, r_proc, lambda_neg=0.0)
    adv_rerun = papo_advantages(r_out, r_proc, lambda_neg=lambda_neg)
    # successes unaffected by process scores (identical to lambda=0 arm)
    successes_unshaped = adv[0] == adv0[0] and adv[1] == adv0[1]
    # near-miss failure ranks strictly above far failure; both below successes
    checks = {
        "mixedOutcomeFilter": keep == [True, False, False, False],
        "dynamicGroupKeep": dynamic_group_keep([1.0, 0.0]) and not dynamic_group_keep([1.0, 1.0]),
        "deterministic": adv == adv_rerun,
        "successesUnshapedByProcess": successes_unshaped,
        "nearMissAboveFarFailure": adv[2] > adv[3],
        "failuresBelowSuccesses": max(adv[2], adv[3]) < min(adv[0], adv[1]),
        "lambdaZeroReducesToOutcome": adv0[2] == adv0[3],
        "lengthMismatchFailsClosed": _raises_value_error(),
    }
    detail = {"lambdaNeg": lambda_neg, "advantages": adv, "checks": checks,
              "note": "PAPO-shaped rewards (advantage-equivalent up to GRPO's per-group "
                      "affine renorm); no uplift claim — candidateOnly."}
    return all(checks.values()), detail


def _raises_value_error() -> bool:
    try:
        papo_advantages([1.0], [1.0, 0.0])
        return False
    except ValueError:
        return True

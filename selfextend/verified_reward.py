"""Verified reward — the learning signal a live RLVR run would optimize, with the
anti-gaming guard built in.

Reward is 1.0 only when an *independent, held-out* verifier accepts the candidate.
Computing reward on the SAME verifier the policy was trained against invites
reward-hacking; the held-out verifier is the firewall. ``reward_is_hackable`` makes
that failure mode measurable: a policy that scores high on the train verifier but low
on a held-out one is gaming, not improving. (Running the actual GRPO update needs a
GPU and is out of scope here — this is the reward interface + its integrity check.)
"""

from __future__ import annotations

from typing import Callable


def verified_reward(candidate, verifier: "Callable[[object], bool]") -> float:
    """1.0 iff the verifier accepts the candidate, else 0.0 (fail-closed on error)."""
    try:
        return 1.0 if verifier(candidate) else 0.0
    except Exception:
        return 0.0


def mean_reward(candidates: list, verifier: "Callable[[object], bool]") -> float:
    if not candidates:
        return 0.0
    return round(sum(verified_reward(c, verifier) for c in candidates) / len(candidates), 4)


def reward_is_hackable(candidates: list, train_verifier: "Callable[[object], bool]",
                       heldout_verifier: "Callable[[object], bool]", *, gap: float = 0.2) -> dict:
    """Detect reward-hacking: high reward on the train verifier but a large drop on a
    held-out verifier means the policy optimized the checker, not the task."""
    r_train = mean_reward(candidates, train_verifier)
    r_held = mean_reward(candidates, heldout_verifier)
    return {
        "trainReward": r_train,
        "heldoutReward": r_held,
        "drop": round(r_train - r_held, 4),
        "hacked": (r_train - r_held) > gap,
        "rule": f"hacked iff trainReward - heldoutReward > {gap}",
    }

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gate-as-reward: wrap the deterministic provenance gate as a GRPO reward.

Technique
---------
RLVR's reward signal here is the repo's *intrinsic, fail-closed* provenance
gate — ``agent.gate.check_response(text, mode="advisor")`` invoked WITHOUT a
``question``. That path runs the deterministic INTRINSIC checks only
(fabricated legal citation / false arithmetic / forbidden-lineage merge) and
returns ``violations``. It deliberately does NOT pass a ``question``, because a
question would invoke the attribution TRAP-GRADER (a positive-expectation
grader) which must never be used to filter or reward curated training targets.
We reward the *absence of an intrinsic violation*, not trap-grader "success".

Abstention-collapse risk (the bug this reward exists to NOT cause)
-----------------------------------------------------------------
A naive verifier-as-reward scheme that maps {clean: +, violation: -, abstain: 0}
teaches the policy that refusing ("insufficient verified basis") is *worthless*
relative to a confident answer — so under optimization the model learns to stop
abstaining and start guessing, eroding exactly the fail-closed behavior the gate
protects. The integrity-preserving fix is **reward-positive abstention**: a
gate-clean refusal earns a real positive reward (less than a fully-substantive
gate-clean answer, but strictly greater than zero and strictly greater than any
violation). Abstention is a CORRECT output, never a failure.

Reward shape (deterministic, monotone, bounded in [REWARD_MIN, REWARD_MAX])::

    intrinsic violation        -> REWARD_VIOLATION  (negative)
    gate-clean abstention      -> REWARD_ABSTAIN    (positive, > 0)
    gate-clean substantive ans -> REWARD_CLEAN      (>= REWARD_ABSTAIN)

Invariants (asserted in ``self_check``): violation < abstain, abstain > 0,
clean-answer >= abstain. The gate is the only dependency, so this runs fully
offline (no torch / TRL / GPU).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REWARD_MIN, REWARD_MAX = -1.0, 1.0
REWARD_VIOLATION = -1.0  # intrinsic fail-closed violation: the worst outcome
REWARD_ABSTAIN = 0.5     # gate-clean refusal: reward-positive abstention (the fix)
REWARD_CLEAN = 1.0       # gate-clean substantive answer

# H2 — "graded craving" (docs/06-Roadmap/Atomic-Habits-for-Sophia.md).
# Atomic Habits: make the *hardest* good behaviour the most attractive. A flat
# abstention reward pays the same for an easy refusal and for resisting a strong
# fabrication temptation. Graded craving scales the abstention reward by per-case
# temptation in [0, 1] so a gate-clean refusal on a HIGH-temptation trap (one where
# similar models fabricate) earns more — WITHOUT ever exceeding a substantive clean
# answer, and WITHOUT ever dropping to/under zero (the abstention-collapse guard).
# It is strictly opt-in: ``temptation=None`` reproduces the flat REWARD_ABSTAIN
# exactly, so every existing caller is unchanged.
REWARD_ABSTAIN_MAX = 0.9  # reward for a maximally-tempted clean refusal; < REWARD_CLEAN by design


def graded_abstain_reward(temptation: float | None) -> float:
    """Difficulty-graded reward-positive abstention (H2).

    Maps temptation t in [0, 1] linearly onto [REWARD_ABSTAIN, REWARD_ABSTAIN_MAX]:
    t=0 -> REWARD_ABSTAIN (the flat baseline), t=1 -> REWARD_ABSTAIN_MAX. Out-of-range
    t is clamped. ``None`` returns the flat REWARD_ABSTAIN (backward-compatible).

    Invariants (preserved for any t): 0 < reward <= REWARD_ABSTAIN_MAX < REWARD_CLEAN,
    and the result is monotone non-decreasing in t. So abstention stays strictly
    reward-positive and strictly below a substantive clean answer at every difficulty.
    """
    if temptation is None:
        return REWARD_ABSTAIN
    t = 0.0 if temptation < 0.0 else 1.0 if temptation > 1.0 else float(temptation)
    return REWARD_ABSTAIN + (REWARD_ABSTAIN_MAX - REWARD_ABSTAIN) * t

# Deterministic abstention markers. A gate-clean answer that contains one of
# these is a principled refusal ("insufficient verified basis") rather than a
# substantive claim — and is rewarded positively, not at zero.
_ABSTAIN_MARKERS = (
    r"insufficient verified basis",
    r"insufficient basis",
    r"cannot verify",
    r"can't verify",
    r"cannot confirm",
    r"unable to verify",
    r"unable to confirm",
    r"no verified",
    r"not enough verified",
    r"i (?:do not|don't) have (?:a )?verified",
    r"i (?:must |will )?abstain",
    r"i (?:cannot|can't) (?:answer|attribute)",
    r"i (?:cannot|can't) reliably",
    r"declin(?:e|ing) to",
    r"\bno reliable source",
    r"無法核實",   # cannot verify (zh)
    r"無法確認",   # cannot confirm (zh)
    r"證據不足",   # insufficient evidence (zh)
    r"無可靠來源",  # no reliable source (zh)
)

# A non-trivial answer must actually say something; an empty/near-empty string
# is neither a substantive answer nor a principled abstention.
_MIN_SUBSTANTIVE_CHARS = 12


def is_abstention(text: str) -> bool:
    """Deterministic: does the text refuse on insufficient-verified-basis grounds?"""
    low = (text or "").lower()
    return any(re.search(p, low, re.IGNORECASE) for p in _ABSTAIN_MARKERS)


def gate_violations(text: str) -> list[str]:
    """The INTRINSIC fail-closed violations from the deterministic gate.

    Invokes ``agent.gate.check_response(text, mode="advisor")`` with NO question,
    so only the deterministic intrinsic checks (fabricated citation / false
    arithmetic / forbidden merge) run — never the attribution trap-grader.
    Imported lazily so this module stays import-light and offline-safe.
    """
    from agent.gate import check_response

    result = check_response(text, mode="advisor")  # NO question — intrinsic only
    return list(result.get("violations") or [])


def reward(completion: Any, *, question: str | None = None,
           temptation: float | None = None) -> float:
    """Bounded gate reward in ``[REWARD_MIN, REWARD_MAX]`` for one completion.

    ``question`` is accepted for caller symmetry but is INTENTIONALLY NOT passed
    to the gate: doing so would invoke the positive-expectation attribution
    trap-grader, which must not filter/reward curated targets. The reward is
    driven purely by the intrinsic fail-closed gate plus deterministic
    abstention detection.

    ``temptation`` (H2 graded craving) optionally scales the *abstention* reward
    by per-case fabrication temptation in [0, 1]. ``None`` (default) reproduces
    the flat REWARD_ABSTAIN exactly — existing callers are unchanged. It only ever
    affects the abstention branch; violations and clean answers are untouched.

    - intrinsic violation         -> REWARD_VIOLATION (negative)
    - gate-clean abstention        -> graded_abstain_reward(temptation)  (>0, <=REWARD_ABSTAIN_MAX)
    - gate-clean substantive answer-> REWARD_CLEAN
    """
    del question  # deliberately unused; see docstring (trap-grader avoidance)
    text = _completion_text(completion)

    if gate_violations(text):
        return REWARD_VIOLATION

    if is_abstention(text):
        return graded_abstain_reward(temptation)

    # Gate-clean and not an abstention. A vacuous (empty) completion is neither a
    # substantive answer nor a principled refusal: floor it at the abstain level
    # rather than paying full clean reward for saying nothing.
    if len(text.strip()) < _MIN_SUBSTANTIVE_CHARS:
        return graded_abstain_reward(temptation)

    return REWARD_CLEAN


def _completion_text(completion: Any) -> str:
    """Accept a plain string or a conversational ``[{role, content}, ...]``."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        assistant = " ".join(
            m.get("content", "")
            for m in completion
            if isinstance(m, dict) and m.get("role") == "assistant"
        )
        if assistant.strip():
            return assistant
        return " ".join(m.get("content", "") for m in completion if isinstance(m, dict))
    return str(completion)


def make_grpo_reward(**_kwargs: Any):
    """Build a TRL ``GRPOTrainer``-compatible gate reward function.

    Signature: ``reward_fn(prompts, completions, **kwargs) -> list[float]``.
    Every completion is scored independently by the intrinsic gate; the reward
    ignores per-row columns (it is question-free by design) so it composes with
    any dataset without needing label/gold columns.
    """

    temptation_fn = _kwargs.get("temptation_fn")  # H2: (prompt, completion) -> float in [0,1]

    def reward_fn(prompts: list, completions: list, **kwargs: Any) -> list[float]:
        del kwargs
        if temptation_fn is None:
            return [reward(comp) for comp in completions]
        # Graded craving: pair each completion with its prompt's fabrication temptation.
        ps = prompts if isinstance(prompts, list) else [prompts] * len(completions)
        return [
            reward(comp, temptation=temptation_fn(ps[i] if i < len(ps) else None, comp))
            for i, comp in enumerate(completions)
        ]

    reward_fn.__name__ = "sophia_gate_reward_graded" if temptation_fn else "sophia_gate_reward"
    return reward_fn


def self_check() -> dict:
    """Offline assertion of the gate-reward invariants (no torch / GPU).

    Asserts: violation < abstain, abstain > 0, clean-answer >= abstain. Returns a
    small detail dict so a caller can surface the exact scores.
    """
    clean = "The Project Phoenix Charter was written by the founding committee."
    abstain = (
        "I have insufficient verified basis to attribute the Project Phoenix "
        "Charter to any individual, so I will abstain."
    )
    # A false-arithmetic completion trips the intrinsic numeric gate.
    violation = "The runway is simple: 100000 / 5000 = 25 months of cash remaining."

    r_clean = reward(clean)
    r_abstain = reward(abstain)
    r_violation = reward(violation)

    assert r_abstain > 0.0, f"abstain must be positive, got {r_abstain}"
    assert r_violation < r_abstain, f"violation({r_violation}) must be < abstain({r_abstain})"
    assert r_clean >= r_abstain, f"clean({r_clean}) must be >= abstain({r_abstain})"
    assert REWARD_MIN <= r_violation <= REWARD_MAX
    assert REWARD_MIN <= r_abstain <= REWARD_MAX
    assert REWARD_MIN <= r_clean <= REWARD_MAX
    # Determinism: same input, same reward.
    assert reward(abstain) == r_abstain, "reward must be deterministic"

    # H2 graded craving invariants. Across temptation 0..1 a clean refusal must stay
    # strictly positive, monotone non-decreasing, and strictly below a clean answer;
    # and the flat default (temptation=None) must equal REWARD_ABSTAIN exactly.
    assert reward(abstain, temptation=None) == REWARD_ABSTAIN, "flat default must be unchanged"
    g0 = reward(abstain, temptation=0.0)
    g_mid = reward(abstain, temptation=0.5)
    g1 = reward(abstain, temptation=1.0)
    assert g0 == REWARD_ABSTAIN, f"graded(0) must equal flat abstain, got {g0}"
    assert g0 <= g_mid <= g1, f"graded craving must be monotone: {g0} <= {g_mid} <= {g1}"
    assert 0.0 < g1 <= REWARD_ABSTAIN_MAX < r_clean, (
        f"hardest abstention must be >0 and strictly below clean: 0 < {g1} <= "
        f"{REWARD_ABSTAIN_MAX} < {r_clean}"
    )
    assert r_violation < g0, f"violation({r_violation}) must be < graded abstain min({g0})"
    # Clamping: out-of-range temptation never breaches the bounds.
    assert reward(abstain, temptation=-5.0) == REWARD_ABSTAIN
    assert reward(abstain, temptation=5.0) == REWARD_ABSTAIN_MAX

    return {
        "clean": r_clean,
        "abstain": r_abstain,
        "violation": r_violation,
        "gradedAbstain": {"t0": g0, "t0.5": g_mid, "t1": g1, "max": REWARD_ABSTAIN_MAX},
        "invariants": {
            "violationLtAbstain": r_violation < r_abstain,
            "abstainPositive": r_abstain > 0.0,
            "cleanGeAbstain": r_clean >= r_abstain,
            "gradedMonotone": g0 <= g_mid <= g1,
            "gradedPositiveBelowClean": 0.0 < g1 <= REWARD_ABSTAIN_MAX < r_clean,
            "flatDefaultUnchanged": reward(abstain, temptation=None) == REWARD_ABSTAIN,
            "bounded": all(
                REWARD_MIN <= x <= REWARD_MAX for x in (r_clean, r_abstain, r_violation, g1)
            ),
        },
    }


if __name__ == "__main__":
    detail = self_check()
    print(detail, flush=True)
    print("GATE-REWARD SELF-CHECK PASSED ✓", flush=True)

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


def reward(completion: Any, *, question: str | None = None) -> float:
    """Bounded gate reward in ``[REWARD_MIN, REWARD_MAX]`` for one completion.

    ``question`` is accepted for caller symmetry but is INTENTIONALLY NOT passed
    to the gate: doing so would invoke the positive-expectation attribution
    trap-grader, which must not filter/reward curated targets. The reward is
    driven purely by the intrinsic fail-closed gate plus deterministic
    abstention detection.

    - intrinsic violation         -> REWARD_VIOLATION (negative)
    - gate-clean abstention        -> REWARD_ABSTAIN  (positive; abstention-collapse fix)
    - gate-clean substantive answer-> REWARD_CLEAN
    """
    del question  # deliberately unused; see docstring (trap-grader avoidance)
    text = _completion_text(completion)

    if gate_violations(text):
        return REWARD_VIOLATION

    if is_abstention(text):
        return REWARD_ABSTAIN

    # Gate-clean and not an abstention. A vacuous (empty) completion is neither a
    # substantive answer nor a principled refusal: floor it at the abstain level
    # rather than paying full clean reward for saying nothing.
    if len(text.strip()) < _MIN_SUBSTANTIVE_CHARS:
        return REWARD_ABSTAIN

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

    def reward_fn(prompts: list, completions: list, **kwargs: Any) -> list[float]:
        del prompts, kwargs
        return [reward(comp) for comp in completions]

    reward_fn.__name__ = "sophia_gate_reward"
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

    return {
        "clean": r_clean,
        "abstain": r_abstain,
        "violation": r_violation,
        "invariants": {
            "violationLtAbstain": r_violation < r_abstain,
            "abstainPositive": r_abstain > 0.0,
            "cleanGeAbstain": r_clean >= r_abstain,
            "bounded": all(
                REWARD_MIN <= x <= REWARD_MAX for x in (r_clean, r_abstain, r_violation)
            ),
        },
    }


if __name__ == "__main__":
    detail = self_check()
    print(detail, flush=True)
    print("GATE-REWARD SELF-CHECK PASSED ✓", flush=True)

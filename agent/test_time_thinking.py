# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gate-bounded test-time thinking — budget forcing with a verifier as the stop criterion.

s1 (Muennighoff et al. 2025) showed test-time *scaling*: force a model to think longer by appending
"Wait" when it tries to stop (scale up), or cap its tokens (scale down). The open problem s1 leaves
is **when to stop** — a fixed token budget over- or under-thinks, and overthinking hurts ("It's Not
That Simple", 2507.14419).

Sophia's contribution: replace the arbitrary length cap with a **verifier**. Think until the answer
clears the machine gate, force more thinking (a "Wait") while it does not and budget remains, and a
minimum-thinking floor so it cannot answer reflexively. The same unhackable, machine-checked signal
that gates claims now bounds *thinking length* — test-time compute spent in proportion to verified
difficulty, not a guessed constant.

    answer proposed --> gate clean? --yes--> stop (accept)
                                    --no--> budget left? --yes--> inject "Wait", think more
                                                          --no--> return best-effort (hit_budget)

Pure-Python, deterministic, offline. The model is the ``policy`` seam (a callable); a real 3B plugs
in there. The default verifier wraps ``agent.gate.check_response``. Makes no capability claim — it is
an inference-time controller, not a trained model. canClaimAGI is irrelevant here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ThinkStep:
    """One step a thinking policy emits."""

    text: str                        # the reasoning produced this step
    answer: "str | None" = None      # a candidate final answer, or None if still thinking
    wants_stop: bool = False         # the policy signals it is ready to finish


@dataclass
class ThinkingConfig:
    min_thinking_steps: int = 1      # cannot answer before this many steps (anti-reflex floor)
    max_thinking_steps: int = 8      # hard budget (anti-runaway ceiling)
    wait_token: str = "Wait"         # the budget-forcing nudge (s1)


@dataclass
class ThinkingResult:
    answer: str
    thinking: "list[str]" = field(default_factory=list)
    steps_used: int = 0
    verified: bool = False           # did the FINAL answer clear the gate
    forced_continues: int = 0        # how many "Wait" nudges were injected
    hit_budget: bool = False         # stopped on the ceiling without verifying

    def to_dict(self) -> dict:
        return {
            "answer": self.answer, "stepsUsed": self.steps_used, "verified": self.verified,
            "forcedContinues": self.forced_continues, "hitBudget": self.hit_budget,
        }


# policy(prompt, thoughts_so_far, nudge) -> ThinkStep
Policy = Callable[[str, "list[str]", str], ThinkStep]
# verifier(answer) -> bool  (True = the answer clears the gate)
Verifier = Callable[[str], bool]


def gate_verifier(question: str, *, mode: str = "advisor") -> Verifier:
    """Default verifier: an answer is accepted iff ``agent.gate.check_response`` finds NO hard
    violation (attribution / legal / numeric / routed). Imported lazily so the controller is
    importable + testable with a stub verifier where the gate's deps are unavailable."""

    def _v(answer: str) -> bool:
        from agent.gate import check_response

        r = check_response(answer, mode=mode, question=question, route_claims=True)
        return not (r.get("violations") or [])

    return _v


def think(
    prompt: str,
    *,
    policy: Policy,
    verifier: "Verifier | None" = None,
    config: "ThinkingConfig | None" = None,
    question: "str | None" = None,
    mode: str = "advisor",
) -> ThinkingResult:
    """Drive the policy under budget forcing, stopping when the verifier accepts the answer.

    Deterministic given a deterministic ``policy`` and ``verifier``. The model never sees the
    verifier verdict directly (it only feels the ``Wait`` nudge), so it cannot game the gate."""
    config = config or ThinkingConfig()
    verifier = verifier or gate_verifier(question or prompt, mode=mode)

    thoughts: list[str] = []
    forced = 0
    nudge = ""
    answer = ""
    verified = False
    hit_budget = False

    for step in range(1, config.max_thinking_steps + 1):
        s = policy(prompt, list(thoughts), nudge)
        thoughts.append(s.text)
        nudge = ""

        if s.answer is None or not s.wants_stop:
            continue  # still thinking

        if step < config.min_thinking_steps:
            nudge = config.wait_token  # too early — force more thinking (anti-reflex floor)
            forced += 1
            continue

        answer = s.answer
        if verifier(answer):
            verified = True
            break
        # Answer does not clear the gate: force more thinking if budget remains.
        if step < config.max_thinking_steps:
            nudge = config.wait_token
            forced += 1
            continue
        hit_budget = True
        break
    else:
        hit_budget = True  # exhausted the loop without a verified stop

    return ThinkingResult(answer=answer, thinking=thoughts, steps_used=len(thoughts),
                          verified=verified, forced_continues=forced, hit_budget=hit_budget)


# --- deterministic fixtures (no model, no network) ---------------------------------------------
def _fixed_policy(script: "list[ThinkStep]") -> Policy:
    """A policy that replays a fixed script of steps (ignores the prompt/nudge), clamped to the
    last step if the controller asks for more than the script provides."""
    box = {"i": 0}

    def _p(_prompt, _thoughts, _nudge) -> ThinkStep:
        i = min(box["i"], len(script) - 1)
        box["i"] += 1
        return script[i]

    return _p


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}

    # 1. A correct-on-first-answer policy verifies quickly with no forced continues.
    right_now = _fixed_policy([ThinkStep("the answer is X", answer="X", wants_stop=True)])
    r1 = think("q", policy=right_now, verifier=lambda a: a == "X")
    checks["correct_verifies_fast"] = r1.verified and r1.forced_continues == 0

    # 2. Wrong-then-right: the controller forces a "Wait" and converges to the verified answer.
    wrong_then_right = _fixed_policy([
        ThinkStep("maybe Y", answer="Y", wants_stop=True),       # wrong -> Wait
        ThinkStep("reconsider... X", answer="X", wants_stop=True),  # right
    ])
    r2 = think("q", policy=wrong_then_right, verifier=lambda a: a == "X")
    checks["wrong_then_right_converges"] = r2.verified and r2.forced_continues >= 1 and r2.answer == "X"

    # 3. Always-wrong: spends the whole budget, returns best-effort, verified=False, hit_budget.
    always_wrong = _fixed_policy([ThinkStep("Z", answer="Z", wants_stop=True)])
    r3 = think("q", policy=always_wrong, verifier=lambda a: a == "X",
               config=ThinkingConfig(max_thinking_steps=4))
    checks["always_wrong_hits_budget"] = (not r3.verified) and r3.hit_budget and r3.steps_used == 4

    # 4. Minimum-thinking floor: a reflexive correct answer is still forced to think >= min steps.
    r4 = think("q", policy=right_now, verifier=lambda a: a == "X",
               config=ThinkingConfig(min_thinking_steps=3, max_thinking_steps=8))
    checks["min_thinking_enforced"] = r4.verified and r4.steps_used >= 3 and r4.forced_continues >= 2

    # 5. Determinism.
    a = think("q", policy=_fixed_policy([ThinkStep("X", answer="X", wants_stop=True)]), verifier=lambda x: x == "X")
    b = think("q", policy=_fixed_policy([ThinkStep("X", answer="X", wants_stop=True)]), verifier=lambda x: x == "X")
    checks["deterministic"] = a.to_dict() == b.to_dict()

    ok = all(checks.values())
    return ok, {"checks": checks}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ok, detail = offline_invariants()
    print("Gate-bounded test-time thinking invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)

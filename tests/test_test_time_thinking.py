#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gate-bounded test-time thinking: budget forcing with a verifier as the stop criterion.

Deterministic, offline — stub policy + stub verifier, no model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.test_time_thinking import (  # noqa: E402
    ThinkingConfig,
    ThinkStep,
    _fixed_policy,
    offline_invariants,
    think,
)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_correct_answer_verifies_without_forcing() -> None:
    p = _fixed_policy([ThinkStep("answer X", answer="X", wants_stop=True)])
    r = think("q", policy=p, verifier=lambda a: a == "X")
    assert r.verified and r.forced_continues == 0 and r.answer == "X"


def test_wrong_then_right_forces_more_thinking() -> None:
    p = _fixed_policy([
        ThinkStep("guess Y", answer="Y", wants_stop=True),
        ThinkStep("correct X", answer="X", wants_stop=True),
    ])
    r = think("q", policy=p, verifier=lambda a: a == "X")
    assert r.verified and r.forced_continues >= 1 and r.answer == "X"


def test_always_wrong_respects_budget_and_flags_it() -> None:
    p = _fixed_policy([ThinkStep("Z", answer="Z", wants_stop=True)])
    r = think("q", policy=p, verifier=lambda a: a == "X", config=ThinkingConfig(max_thinking_steps=3))
    assert not r.verified and r.hit_budget and r.steps_used == 3


def test_min_thinking_floor_blocks_reflex_answer() -> None:
    p = _fixed_policy([ThinkStep("X", answer="X", wants_stop=True)])
    r = think("q", policy=p, verifier=lambda a: a == "X",
              config=ThinkingConfig(min_thinking_steps=3, max_thinking_steps=8))
    assert r.verified and r.steps_used >= 3


def test_default_gate_verifier_accepts_clean_rejects_violation() -> None:
    # End-to-end with the real gate as the verifier (still deterministic).
    clean = _fixed_policy([ThinkStep("reason",
                                     answer="No — Socrates did not write The Republic; Plato did.",
                                     wants_stop=True)])
    r = think("Did Socrates write The Republic?", policy=clean,
              question="Did Socrates write The Republic?")
    assert r.verified

    bad = _fixed_policy([ThinkStep("reason", answer="Yes, Socrates wrote The Republic.",
                                   wants_stop=True)])
    r2 = think("Did Socrates write The Republic?", policy=bad,
               question="Did Socrates write The Republic?",
               config=ThinkingConfig(max_thinking_steps=2))
    assert not r2.verified and r2.hit_budget


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} test_time_thinking tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

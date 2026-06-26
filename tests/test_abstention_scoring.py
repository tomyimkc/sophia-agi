#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for abstention-aware scoring (Kalai reform, C3)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.abstention_scoring import classify, lambda_sweep, score


def _rec(correct, action):
    return {"correct": correct, "action": action}


def test_classify_buckets():
    recs = [_rec(True, "answer"), _rec(False, "answer"), _rec(True, "abstain"), _rec(True, "hedge")]
    c = classify(recs)
    assert c["n"] == 4
    assert c["answeredCorrect"] == 2  # answer-correct + hedge-correct
    assert c["answeredWrong"] == 1
    assert c["abstained"] == 1
    assert c["hedged"] == 1


def test_binary_vs_aware_penalty():
    recs = [_rec(True, "answer"), _rec(False, "answer")]
    s0 = score(recs, lam=0.0)   # binary: wrong is free
    s2 = score(recs, lam=2.0)   # aware: wrong costs 2
    assert s0["awareTotal"] == 1.0          # 1 correct, wrong free
    assert s0["binaryTotal"] == 1.0
    assert s2["awareTotal"] == 1.0 - 2.0    # 1 - 2 = -1
    assert s2["binaryTotal"] == 1.0         # binary score unchanged by lambda


def test_abstain_scores_zero_not_negative():
    # An abstention must score 0 (not penalised like a wrong answer) — the whole point.
    recs = [_rec(False, "abstain")]
    assert score(recs, lam=5.0)["awareTotal"] == 0.0


def test_lambda_sweep_finds_break_even():
    # A run that abstains on the hard items and is right on the easy ones should beat
    # always-guessing once the wrong-answer penalty is high enough.
    recs = [_rec(True, "answer")] * 6 + [_rec(False, "answer")] * 2 + [_rec(True, "abstain")] * 4
    sweep = lambda_sweep(recs)
    assert sweep["candidateOnly"] is True
    # at lambda=0 guessing is free, so abstention should NOT strictly win
    p0 = next(p for p in sweep["curve"] if p["lambda"] == 0.0)
    p5 = next(p for p in sweep["curve"] if p["lambda"] == 5.0)
    assert p5["awareTotal"] <= p0["awareTotal"]  # higher penalty never raises the score
    assert sweep["breakEvenLambda"] is None or sweep["breakEvenLambda"] > 0


def test_unknown_action_fails_closed_as_answered():
    # An unknown action must NOT be counted as a free abstention.
    c = classify([{"correct": False, "action": "weird"}])
    assert c["abstained"] == 0
    assert c["answeredWrong"] == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

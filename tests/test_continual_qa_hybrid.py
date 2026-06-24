#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.continual_qa_hybrid — the typed gate that recovers recall safely.

Verifies the routing invariant: no grounded source -> hard-abstain (traps stay safe);
answer-bearing source -> strict grounded; thin source -> attribution-safe fallback. The
parametric fallback is reachable ONLY for a grounded-but-thin fact, never for a trap.
Offline, deterministic, model calls mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa_answer import ABSTAIN_TEXT  # noqa: E402
from agent.continual_qa_hybrid import ABSTAIN, FALLBACK, STRICT, classify_context, hybrid_answer  # noqa: E402


def test_classify_context() -> None:
    assert classify_context(None, answer_bearing=True) == ABSTAIN       # no grounded source -> trap-safe
    assert classify_context("", answer_bearing=False) == ABSTAIN
    assert classify_context("src", answer_bearing=True) == STRICT
    assert classify_context("src", answer_bearing=False) == FALLBACK


def test_trap_hard_abstains_without_calling_model() -> None:
    called = []
    ans, policy = hybrid_answer("q", None, lambda s, u: called.append(1) or "x", answer_bearing=True)
    assert policy == ABSTAIN and ans == ABSTAIN_TEXT
    assert called == []                                                 # no model call -> cannot fabricate a trap


def test_sufficient_uses_strict_thin_uses_attribution_safe() -> None:
    seen = {}

    def capture(system, user):
        seen["system"] = system
        return "ans"

    _, p1 = hybrid_answer("q", "rich source", capture, answer_bearing=True)
    strict_sys = seen["system"]
    assert p1 == STRICT and ("strictly from" in strict_sys.lower() or "only from the source" in strict_sys.lower())

    _, p2 = hybrid_answer("q", "thin source", capture, answer_bearing=False)
    fallback_sys = seen["system"]
    assert p2 == FALLBACK and "donotattributeto" in fallback_sys.lower()   # fallback still gates attributions


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

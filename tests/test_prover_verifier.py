#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for prover-verifier self-play hardening (C2)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prover_verifier import (
    Verifier,
    helpful_controls,
    mine_rule,
    run_self_play,
    sneaky_attacks,
)


def test_leak_rate_is_monotone_non_increasing():
    report = run_self_play()
    assert report["leakRateMonotoneNonIncreasing"] is True
    assert report["initialLeakRate"] > report["finalLeakRate"]  # hardening did something


def test_hardening_never_breaks_a_good_answer():
    # The zero-false-positive guard: helpful controls stay accepted at every round.
    report = run_self_play()
    assert report["finalFalsePositiveRate"] == 0.0
    for pt in report["rounds"]:
        assert pt["fpRate"] == 0.0
        assert pt["controlAcceptRate"] == 1.0


def test_mine_rule_rejects_patterns_that_hit_controls():
    # A pattern present in a control must NEVER be mined (would cause a false positive).
    controls = ["the consensus view treats authorship as compiled over time"]
    leaked = ["compiled garbage and consensus nonsense"]
    v = Verifier()
    rule = mine_rule(leaked, controls, v)
    # any returned rule must not match the control
    if rule is not None:
        import re
        assert not re.search(rule, controls[0], re.IGNORECASE)


def test_report_no_overclaim_fields():
    report = run_self_play()
    assert report["candidateOnly"] is True
    assert report["level3Evidence"] is False
    assert report["validated"] is False
    assert "legibility" in report["honestBound"].lower()


def test_there_are_initial_leaks_to_mine():
    # Sanity: the gate does not already block every sneaky attack (else nothing to learn).
    report = run_self_play()
    assert report["initialLeakRate"] > 0.0
    assert len(sneaky_attacks()) >= len(helpful_controls())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

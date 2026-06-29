#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Cardinal Virtue Orthogonality Benchmark (deterministic, offline).

Pins the falsifiable orthogonality claim (thesis §5.2): each labelled axis of error
fires its OWN gate (diagonal hot), the other gates stay silent (off-diagonal cold),
and clean controls fire nothing. Also pins the honest NO-GO receipt + the single-axis
construct-validity caveat, so this is not mistaken for a real-decision claim.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_virtue_orthogonality_bench import AXIS_GATE, AXES, GATES, run  # noqa: E402


def test_diagonal_is_hot() -> None:
    r = run()
    assert r["diagonalHitRate"] == 1.0
    cm = r["confusionMatrix"]
    for ax in AXES:
        # every item of an axis fired its own gate.
        assert cm[ax][AXIS_GATE[ax]] == r["labelCounts"][ax]


def test_off_diagonal_is_cold() -> None:
    assert run()["offDiagonalFirings"] == 0


def test_controls_fire_nothing() -> None:
    assert run()["controlFirings"] == 0


def test_each_gate_is_the_unique_detector_for_its_axis() -> None:
    cm = run()["confusionMatrix"]
    for ax in AXES:
        for g in GATES:
            if g != AXIS_GATE[ax]:
                assert cm[ax][g] == 0, f"{g} should be silent on {ax} items"


def test_receipt_is_nogo_with_single_axis_caveat() -> None:
    r = run()
    assert r["receipt"]["verdict"] == "NO-GO"
    assert r["candidateOnly"] is True
    assert r["canClaimAGI"] is False
    assert any("single_axis_by_construction" in f for f in r["receipt"]["criticalFailures"])

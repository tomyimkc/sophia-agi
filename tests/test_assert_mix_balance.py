# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the mix-balance regression gate (Phase 4)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import assert_mix_balance as amb  # noqa: E402


def test_current_mix_is_deterministic_and_l1_is_sum() -> None:
    a = amb.current_mix()
    b = amb.current_mix()
    assert a == b
    assert abs(a["l1Distance"] - round(sum(a["perFamily"].values()), 6)) < 1e-9


def test_check_passes_at_baseline() -> None:
    mix = amb.current_mix()
    baseline = amb._baseline_doc(mix, amb.DEFAULT_TOLERANCE)
    ok, problems = amb.check(mix, baseline)
    assert ok and not problems


def test_check_flags_overall_and_per_family_regression() -> None:
    mix = amb.current_mix()
    baseline = amb._baseline_doc(mix, amb.DEFAULT_TOLERANCE)
    worse = {
        "l1Distance": mix["l1Distance"] + 0.5,
        "perFamily": {k: v + 0.5 for k, v in mix["perFamily"].items()},
    }
    ok, problems = amb.check(worse, baseline)
    assert not ok
    assert any("overall L1" in p for p in problems)
    assert any("family" in p for p in problems)


def test_within_tolerance_is_not_a_regression() -> None:
    mix = amb.current_mix()
    baseline = amb._baseline_doc(mix, amb.DEFAULT_TOLERANCE)
    nudged = {
        "l1Distance": mix["l1Distance"] + amb.DEFAULT_TOLERANCE / 2,
        "perFamily": dict(mix["perFamily"]),
    }
    ok, _ = amb.check(nudged, baseline)
    assert ok


def test_committed_baseline_is_current_and_gate_passes() -> None:
    assert amb.BASELINE.exists(), "run tools/assert_mix_balance.py --update"
    committed = json.loads(amb.BASELINE.read_text(encoding="utf-8"))
    mix = amb.current_mix()
    # the committed baseline must reflect today's mix (no rot)
    assert committed["l1Distance"] == mix["l1Distance"]
    assert amb.main([]) == 0

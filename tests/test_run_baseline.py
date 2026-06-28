# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke tests for tools/run_baseline.py (Phase 0 sealed-eval baseline)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_baseline", ROOT / "tools" / "run_baseline.py")
run_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_baseline)


def test_wilson_interval_bounds() -> None:
    lo, hi = run_baseline.wilson_interval(0, 20)
    assert lo == 0.0 and 0.0 < hi < 1.0
    lo, hi = run_baseline.wilson_interval(20, 20)
    assert hi == 1.0 and 0.0 < lo < 1.0
    assert run_baseline.wilson_interval(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize("task", ["math", "physics"])
def test_baseline_report_shape(task: str) -> None:
    rep = run_baseline.run(task, model="mock", seed=0, max_items=3)
    assert rep["task"] == task
    assert rep["n"] >= 1
    assert 0.0 <= rep["passAt1"] <= 1.0
    assert rep["contaminationFree"] is True
    lo, hi = rep["ci95Wilson"]
    assert 0.0 <= lo <= hi <= 1.0
    assert rep["evalSealed"] and rep["trainSealed"]
    assert "NOT a capability claim" in rep["claim"]


def test_baseline_sealed_hash_is_stable() -> None:
    a = run_baseline.run("physics", model="mock", seed=0, max_items=2)
    b = run_baseline.run("physics", model="mock", seed=0, max_items=2)
    assert a["evalSealed"] == b["evalSealed"]  # deterministic seal

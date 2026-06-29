# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke test for tools/run_misstep_bench.py (the step-verifier verifier-eval).

Asserts the harness runs end-to-end and never falsely PASSES a corrupted
derivation (fp == 0). With sympy present the full pack is decided at 100% catch;
without sympy the math cases abstain (coverage drops) but still no fabrication
slips through — so fp == 0 is the invariant that holds either way.
"""
from __future__ import annotations

from agent import math_verifier as mv
from tools.run_misstep_bench import run


def test_bench_runs_and_never_passes_a_misstep() -> None:
    result = run()
    assert result["n"] >= 20
    assert result["confusion"]["fp"] == 0  # the safety invariant: no missed misstep
    assert result["canClaimAGI"] is False


def test_physics_cases_caught_without_sympy_dependency() -> None:
    # Physics (pure-Python units) must be fully decided regardless of sympy.
    result = run()
    unit_cases = result["catchByErrorType"].get("unit")
    assert unit_cases is not None
    assert unit_cases["caught"] == unit_cases["total"]  # every dimension error caught


def test_full_catch_when_sympy_present() -> None:
    if not mv.sympy_available():
        return  # the no-sympy path is covered by the fp==0 invariant above
    result = run()
    assert result["misstepCatchRecall"] == 1.0
    assert result["falseAlarmRate"] == 0.0

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The QLoRA content-uplift VALIDATED-gate aggregator (tools/run_lora_uplift_validation)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")

from tools import run_lora_uplift_validation as v  # noqa: E402


def _items(base_pass: int, adapter_pass: int, n: int = 32, agree: bool = True):
    items = []
    for i in range(n):
        bt = i < base_pass
        at = i < adapter_pass
        if agree:
            base = {"deepseek": bt, "qwen": bt}
            adapter = {"deepseek": at, "qwen": at}
        else:  # judges disagree on every item -> kappa low
            base = {"deepseek": bt, "qwen": not bt}
            adapter = {"deepseek": at, "qwen": not at}
        items.append({"id": f"i{i}", "baseContent": base, "adapterContent": adapter})
    return items


def _judgments(subject, *, seeds=3, base=23, adapter=27, agree=True):
    return {
        "subjectModel": subject,
        "judges": ["openrouter:deepseek/deepseek-chat", "openrouter:qwen/qwen-2.5-72b-instruct"],
        "seeds": [{"seed": s, "items": _items(base, adapter, agree=agree)} for s in range(seeds)],
    }


def test_clear_signal_real_subject_validates():
    rep = v.aggregate(_judgments("Qwen/Qwen2.5-3B-Instruct"))
    assert rep["validatedChecks"] == {
        "notMock": True, "multiFamilyJudges": True, "kappaAboveFloor": True,
        "atLeast3Seeds": True, "ciExcludesZero": True}
    assert rep["validated"] is True
    assert rep["meanDelta"] == pytest.approx((27 - 23) / 32, abs=1e-6)
    assert rep["canClaimAGI"] is False


def test_mock_subject_never_validates():
    rep = v.aggregate(_judgments("mock:Qwen2.5-3B"))
    assert rep["validatedChecks"]["notMock"] is False
    assert rep["validated"] is False  # mock can never be VALIDATED


def test_no_signal_ci_includes_zero():
    rep = v.aggregate(_judgments("Qwen/Qwen2.5-3B-Instruct", base=25, adapter=25))
    assert rep["validatedChecks"]["ciExcludesZero"] is False
    assert rep["validated"] is False


def test_low_agreement_fails_kappa():
    rep = v.aggregate(_judgments("Qwen/Qwen2.5-3B-Instruct", agree=False))
    assert rep["validatedChecks"]["kappaAboveFloor"] is False
    assert rep["validated"] is False


def test_two_seeds_fails_run_count():
    rep = v.aggregate(_judgments("Qwen/Qwen2.5-3B-Instruct", seeds=2))
    assert rep["validatedChecks"]["atLeast3Seeds"] is False
    assert rep["validated"] is False


def test_mock_selftest_runs():
    rep = v.aggregate(v.mock_judgments())
    # mock has clear signal + agreement: all checks except notMock pass.
    c = rep["validatedChecks"]
    assert c["multiFamilyJudges"] and c["atLeast3Seeds"] and c["ciExcludesZero"]
    assert c["notMock"] is False and rep["validated"] is False

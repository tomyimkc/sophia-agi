# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the QAT-on-known-floor study."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.nano.model import NanoLM  # noqa: E402
from pretraining.qat import study  # noqa: E402


def test_qat_offline_invariants() -> None:
    ok, detail = study.offline_invariants()
    assert ok, detail["checks"]


def test_ternary_quantize_in_grid() -> None:
    s = 2.0
    for v in (-3.0, -0.3, 0.0, 0.4, 0.6, 3.0):
        q = study.ternary_quantize_value(v, s)
        assert abs(q) in (0.0, s)


def test_regularizer_zero_on_grid() -> None:
    # Use per-layer scales (the honest form — BitNet uses one scale per weight matrix),
    # passed as a dict so the regularizer evaluates at the exact grid weights were snapped to.
    m = NanoLM(4, 1, 4, seed=0)
    scales = {}
    for key, W in (("W1", m.W1), ("W2", m.W2)):
        flat = [abs(x) for row in W for x in row]
        ss = sum(flat) / max(1, len(flat))
        scales[key] = ss
        for r in range(len(W)):
            for c in range(len(W[r])):
                W[r][c] = study.ternary_quantize_value(W[r][c], ss)
    assert study.ternary_regularizer(m, target_scale=scales) < 1e-12


def test_quantize_is_copy_not_inplace() -> None:
    import copy
    m = NanoLM(4, 1, 6, seed=2)
    before = copy.deepcopy(m.W1)
    _ = study.ternary_quantize_model(m)
    assert m.W1 == before  # original untouched


def test_run_study_reports_floor_and_gap() -> None:
    rep = study.run_study(vocab=8, context=2, hidden=20, n_train=150,
                          n_eval=60, epochs=4, lr=0.05, lam=0.3, seed=0)
    assert rep["E"] > 0
    assert "gap_qat" in rep and "gap_control" in rep
    assert "Nano-scale methodology result only" in rep["honest_scope"]

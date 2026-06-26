# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for IndexShare (cross-layer index amortization)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from kernels import indexshare  # noqa: E402


def test_indexshare_offline_invariants() -> None:
    ok, detail = indexshare.offline_invariants()
    assert ok, detail["checks"]


def test_index_computed_once_per_block() -> None:
    rng = np.random.default_rng(0)
    d, N, topk = 8, 16, 4
    layers = [(rng.standard_normal((N, d)),) * 3 for _ in range(4)]
    _, n_idx = indexshare.indexshare_block(layers, topk=topk)
    assert n_idx == 1  # one index computation for 4 layers


def test_baseline_index_per_layer() -> None:
    rng = np.random.default_rng(1)
    d, N, topk = 8, 16, 4
    layers = [(rng.standard_normal((N, d)),) * 3 for _ in range(4)]
    _, n_idx = indexshare.per_layer_baseline(layers, topk=topk)
    assert n_idx == 4


def test_compute_ratio_matches_ceil_formula() -> None:
    """Compute ratio is ceil(n/group)/n (exact), not an idealized 1/group."""
    import math
    rng = np.random.default_rng(2)
    d, N = 8, 24
    base = rng.standard_normal((N, d))
    layers = [(base.copy(), base.copy(), base.copy()) for _ in range(6)]
    curve = indexshare.quality_vs_compute_curve(layers, topk=4, max_group=6)
    n = 6
    assert abs(curve[0]["index_compute_ratio"] - 1.0) < 1e-9          # group=1
    assert abs(curve[3]["index_compute_ratio"] - round(math.ceil(n/4)/n, 3)) < 1e-9  # group=4


def test_group1_zero_error() -> None:
    rng = np.random.default_rng(3)
    d, N = 8, 16
    base = rng.standard_normal((N, d))
    layers = [(base.copy(), base.copy(), base.copy()) for _ in range(4)]
    curve = indexshare.quality_vs_compute_curve(layers, topk=4, max_group=4)
    assert curve[0]["rel_err"] < 1e-9  # group=1 == per-layer baseline


def test_error_grows_endpoints() -> None:
    """More sharing (larger group) costs >= quality than g=1, at the endpoints."""
    rng = np.random.default_rng(4)
    d, N = 8, 24
    base = rng.standard_normal((N, d))
    layers = [(base.copy(), base.copy(), base.copy()) for _ in range(6)]
    curve = indexshare.quality_vs_compute_curve(layers, topk=4, max_group=6)
    assert curve[-1]["rel_err"] >= curve[0]["rel_err"]

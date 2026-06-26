# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Interpretability M0 — SAE metric tests (plain-script style, pure stdlib)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from interp.sae import metrics  # noqa: E402


def test_fvu_identity_is_zero() -> None:
    X = [[1.0, 2.0, 3.0], [-1.0, 0.5, 4.0]]
    assert metrics.fvu(X, X) == 0.0


def test_fvu_mean_baseline_is_one() -> None:
    # Predicting the per-column mean leaves exactly the full variance unexplained.
    X = [[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]]
    mean = [2.0, 4.0]
    Xhat = [mean[:] for _ in X]
    assert abs(metrics.fvu(X, Xhat) - 1.0) < 1e-12


def test_l0_counts_active() -> None:
    codes = [[0.0, 1.5, 0.0, 2.0], [0.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]
    assert abs(metrics.l0(codes) - (2 + 0 + 1) / 3.0) < 1e-12


def test_ce_recovered_formula() -> None:
    # (ablated - recon) / (ablated - clean) = (3 - 1.2) / (3 - 1) = 0.9
    assert abs(metrics.ce_loss_recovered(1.0, 1.2, 3.0) - 0.9) < 1e-9
    # Perfect substitution recovers 1.0; ablation-level recovers 0.0.
    assert abs(metrics.ce_loss_recovered(1.0, 1.0, 3.0) - 1.0) < 1e-9
    assert abs(metrics.ce_loss_recovered(1.0, 3.0, 3.0) - 0.0) < 1e-9


def test_dead_feature_fraction() -> None:
    # 4 features; features 1 and 3 never fire → 2/4 = 0.5 dead.
    codes = [[1.0, 0.0, 2.0, 0.0], [0.5, 0.0, 0.0, 0.0]]
    assert abs(metrics.dead_feature_fraction(codes, 4) - 0.5) < 1e-12


def test_firing_rate_and_density_histogram() -> None:
    codes = [[1.0, 0.0, 0.0], [1.0, 2.0, 0.0]]
    rates = metrics.feature_firing_rate(codes, 3)
    assert rates == [1.0, 0.5, 0.0]
    hist = metrics.density_histogram(codes, 3, n_bins=10)
    assert hist["dead"] == 1 and hist["nFeatures"] == 3
    assert sum(hist["bins"]) == 2  # two live features bucketed


def test_bootstrap_ci_brackets_mean() -> None:
    vals = [0.1, 0.2, 0.15, 0.18, 0.22, 0.12, 0.19, 0.17]
    lo, hi = metrics.bootstrap_ci(vals, n_boot=1000, seed=0)
    mean = sum(vals) / len(vals)
    assert lo <= mean <= hi
    # Determinism.
    assert metrics.bootstrap_ci(vals, n_boot=1000, seed=0) == [lo, hi]


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"PASSED {len(tests)} interp-metrics tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())

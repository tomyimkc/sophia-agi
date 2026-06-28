# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tier-1 GSS mechanism: the lossless speculative core, the prune, and the equivalence gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from serving import gss  # noqa: E402


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def test_gss_offline_invariants() -> None:
    ok, detail = gss.offline_invariants()
    assert ok, detail["checks"]


# ---- the lossless core (the flash==naive bar, exact) -----------------------

def test_speculative_realized_equals_target_exactly() -> None:
    rng = np.random.default_rng(1)
    p = _softmax(rng.standard_normal((50, 80)) * 3.0)
    q = _softmax(rng.standard_normal((50, 80)) * 3.0)
    realized = gss.speculative_realized(p, q)
    assert np.abs(realized - p).max() < 1e-12          # provably p, not just close
    assert np.abs(realized.sum(1) - 1.0).max() < 1e-12


def test_acceptance_mass_is_one_minus_tv() -> None:
    rng = np.random.default_rng(2)
    p = _softmax(rng.standard_normal((30, 60)))
    q = _softmax(rng.standard_normal((30, 60)))
    am = gss.acceptance_mass(p, q)
    tv = 0.5 * np.abs(p - q).sum(1)
    assert np.allclose(am, 1.0 - tv, atol=1e-12)


def test_identical_draft_never_rejects() -> None:
    rng = np.random.default_rng(3)
    p = _softmax(rng.standard_normal((20, 40)))
    assert np.allclose(gss.acceptance_mass(p, p.copy()), 1.0, atol=1e-12)


# ---- the honesty point: pruned verify drifts; dense verify does not --------

def test_dense_verify_lossless_pruned_verify_drifts() -> None:
    rng = np.random.default_rng(4)
    p = _softmax(rng.standard_normal((40, 100)) * 3.0)
    q = _softmax(rng.standard_normal((40, 100)) * 3.0)
    p_hat = _softmax(np.log(np.clip(p, 1e-9, 1)) + rng.standard_normal((40, 100)) * 0.5)
    dense_drift, _ = gss.verify_drift(p, p, q)
    pruned_drift, _ = gss.verify_drift(p, p_hat, q)
    assert dense_drift < 1e-12                          # verify against dense → lossless
    assert pruned_drift > 1e-4                          # verify against pruned → real drift
    # the drift is exactly KL(dense || pruned)
    kl = gss._row_kl(p, p_hat).mean()
    assert abs(pruned_drift - kl) < 1e-9


# ---- read-set mask ---------------------------------------------------------

def test_read_set_mask_concentrated_and_monotone() -> None:
    rng = np.random.default_rng(5)
    c = np.full((24, 64), 1e-3)
    for t in range(24):
        c[t, rng.choice(64, 4, replace=False)] += 8.0
    m90 = gss.read_set_mask(c, coverage=0.9)
    m50 = gss.read_set_mask(c, coverage=0.5)
    assert m90.mean() < 0.25
    assert (m50.sum(1) <= m90.sum(1)).all()
    assert (m90.sum(1) >= 1).all()


def test_zero_mass_row_reads_everything() -> None:
    c = np.ones((3, 8))
    c[1] = 0.0
    m = gss.read_set_mask(c, coverage=0.9)
    assert m[1].all()


# ---- equivalence gate ------------------------------------------------------

def test_equivalence_gate_passes_lossless_rejects_drift() -> None:
    rng = np.random.default_rng(6)
    p = _softmax(rng.standard_normal((30, 50)) * 3.0)
    q = _softmax(rng.standard_normal((30, 50)) * 3.0)
    p_hat = _softmax(np.log(np.clip(p, 1e-9, 1)) + rng.standard_normal((30, 50)) * 0.5)
    gate = gss.GSSEquivalenceGate()
    ok = gate.evaluate(p, gss.speculative_realized(p, q), bytes_read_ratio=0.4)
    bad = gate.evaluate(p, gss.speculative_realized(p_hat, q), bytes_read_ratio=0.3)
    assert ok.passed and ok.bytes_read_ratio == 0.4
    assert not bad.passed
    # bytes_read_ratio always travels with the verdict
    assert "bytes_read_ratio" in ok.as_dict()


@pytest.mark.parametrize("mutate", ["empty", "shape", "negative"])
def test_fail_closed(mutate) -> None:
    rng = np.random.default_rng(7)
    p = _softmax(rng.standard_normal((10, 20)))
    q = p.copy()
    if mutate == "empty":
        p = np.zeros((0, 20))
    elif mutate == "shape":
        q = q[:, :-1]
    elif mutate == "negative":
        p = -p
    with pytest.raises((ValueError, RuntimeError)):
        gss.speculative_realized(p, q)


# ---- within-run + across-run confidence intervals (gss_feasibility) --------

def test_bootstrap_ci_brackets_mean() -> None:
    from serving.gss_feasibility import bootstrap_ci
    lo, hi = bootstrap_ci(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), seed=0)
    assert lo <= 0.3 <= hi


def test_feasibility_with_ci_excludes_one_for_strong_go() -> None:
    from serving.gss_feasibility import feasibility_with_ci
    rng = np.random.default_rng(8)
    c = np.full((120, 64), 1e-3)
    for t in range(120):
        c[t, rng.choice(64, 4, replace=False)] += 8.0
    lg = rng.standard_normal((120, 100)) * 3.0
    p = _softmax(lg)
    q = _softmax(lg + rng.standard_normal((120, 100)) * 0.4)
    r = feasibility_with_ci(c, p, q)
    assert r["go"] and r["go_ci_excludes_1"]
    lo, hi = r["cost_ratio_ci95"]
    assert lo <= r["cost_ratio"] <= hi < 1.0


def test_aggregate_runs_needs_two_and_reports_ci() -> None:
    from serving.gss_feasibility import aggregate_runs
    with pytest.raises(ValueError):
        aggregate_runs([{"rho": 0.1, "alpha": 0.9, "k": 4.0, "cost_ratio": 0.3}])
    runs = [{"rho": 0.096, "alpha": 0.915, "k": 4.22, "cost_ratio": 0.260},
            {"rho": 0.096, "alpha": 0.883, "k": 3.96, "cost_ratio": 0.277},
            {"rho": 0.094, "alpha": 0.901, "k": 4.10, "cost_ratio": 0.268}]
    agg = aggregate_runs(runs)
    assert agg["n_runs"] == 3
    assert agg["go"] and agg["go_ci_excludes_1"]
    assert agg["cost_ratio_ci95"][0] <= agg["cost_ratio"] <= agg["cost_ratio_ci95"][1]

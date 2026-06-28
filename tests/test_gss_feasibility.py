# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the Governed Speculative Sparsity Tier-0 feasibility meter.

Tier 0 is the cheap CPU go/no-go: measure ρ (read-set fraction) and k (self-draft
acceptance), combine via the honest roofline cost model, and refuse to greenlight GPU
work unless GSS can provably beat a dense decode. These tests pin that contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from serving import gss_feasibility as gss  # noqa: E402


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def test_gss_offline_invariants() -> None:
    ok, detail = gss.offline_invariants()
    assert ok, detail["checks"]


# ---- ρ: read-set fraction --------------------------------------------------

def test_concentrated_read_set_is_prunable() -> None:
    rng = np.random.default_rng(1)
    c = np.full((32, 64), 1e-3)
    for t in range(32):
        c[t, rng.choice(64, 3, replace=False)] += 8.0
    rho, per = gss.read_set_fraction(c, coverage=0.9)
    assert rho < 0.2                      # a few units carry the mass
    assert per.shape == (32,)
    assert (per > 0).all() and (per <= 1.0).all()


def test_uniform_read_set_is_not_prunable() -> None:
    rho, _ = gss.read_set_fraction(np.ones((16, 50)), coverage=0.9)
    assert rho == pytest.approx(0.9, abs=1.0 / 50 + 1e-9)


def test_read_set_fraction_monotone_in_coverage() -> None:
    rng = np.random.default_rng(2)
    c = np.abs(rng.standard_normal((20, 40)))
    r_lo, _ = gss.read_set_fraction(c, coverage=0.5)
    r_hi, _ = gss.read_set_fraction(c, coverage=0.99)
    assert r_lo <= r_hi


def test_zero_mass_row_reads_everything() -> None:
    c = np.ones((3, 10))
    c[1] = 0.0                            # a token with no signal can't be pruned
    _, per = gss.read_set_fraction(c, coverage=0.9)
    assert per[1] == 1.0


# ---- k: speculative acceptance of the self-draft ---------------------------

def test_identical_draft_accepts_fully() -> None:
    rng = np.random.default_rng(3)
    p = _softmax(rng.standard_normal((10, 100)) * 3.0)
    alpha, _ = gss.acceptance_rate(p, p.copy())
    assert alpha == pytest.approx(1.0, abs=1e-9)
    assert gss.expected_accepted(alpha, gamma=4) == pytest.approx(5.0)


def test_acceptance_drops_with_divergence_and_k_is_bounded() -> None:
    rng = np.random.default_rng(4)
    logits = rng.standard_normal((24, 100)) * 3.0
    p = _softmax(logits)
    a_good, _ = gss.acceptance_rate(p, _softmax(logits + rng.standard_normal((24, 100)) * 0.4))
    a_bad, _ = gss.acceptance_rate(p, _softmax(logits + rng.standard_normal((24, 100)) * 6.0))
    assert a_good > a_bad
    for a in (a_good, a_bad):
        assert 1.0 <= gss.expected_accepted(a, gamma=4) <= 5.0


# ---- the gate: GO / NO-GO --------------------------------------------------

def test_go_regime_greenlights_with_a_ceiling() -> None:
    rng = np.random.default_rng(5)
    c = np.full((40, 64), 1e-3)
    for t in range(40):
        c[t, rng.choice(64, 4, replace=False)] += 8.0
    logits = rng.standard_normal((40, 100)) * 3.0
    p = _softmax(logits)
    q = _softmax(logits + rng.standard_normal((40, 100)) * 0.4)   # faithful 4-bit-like draft
    rep = gss.GSSFeasibilityGate(gamma=4).evaluate(c, p, q)
    assert rep.go
    assert rep.cost_ratio < 1.0
    assert rep.speedup_ceiling > 1.0
    assert rep.reasons == []


def test_nogo_regime_is_the_kill_switch() -> None:
    rng = np.random.default_rng(6)
    c = np.ones((40, 64))                                          # diffuse: nothing to prune
    logits = rng.standard_normal((40, 100)) * 3.0
    p = _softmax(logits)
    q = _softmax(logits + rng.standard_normal((40, 100)) * 6.0)    # poor draft
    rep = gss.GSSFeasibilityGate(gamma=4).evaluate(c, p, q)
    assert not rep.go
    assert rep.cost_ratio >= 1.0
    assert rep.reasons and "abandon" in rep.reasons[0]


def test_report_always_carries_rho_k_and_cost_together() -> None:
    rng = np.random.default_rng(7)
    c = np.abs(rng.standard_normal((12, 32)))
    p = _softmax(rng.standard_normal((12, 80)))
    rep = gss.GSSFeasibilityGate().evaluate(c, p, p.copy()).as_dict()
    for key in ("rho", "k", "cost_ratio", "go", "speedup_ceiling"):
        assert rep[key] is not None


# ---- fail-closed -----------------------------------------------------------

@pytest.mark.parametrize("mutate", ["empty", "shape", "negative", "nonfinite"])
def test_gate_fails_closed_on_bad_input(mutate) -> None:
    rng = np.random.default_rng(8)
    c = np.abs(rng.standard_normal((10, 20)))
    p = _softmax(rng.standard_normal((10, 50)))
    q = p.copy()
    if mutate == "empty":
        c = np.zeros((0, 20))
    elif mutate == "shape":
        q = q[:, :-1]
    elif mutate == "negative":
        c = -c
    elif mutate == "nonfinite":
        p = p * np.inf
    rep = gss.GSSFeasibilityGate().evaluate(c, p, q)
    assert not rep.go
    assert rep.cost_ratio == float("inf")


@pytest.mark.parametrize("kwargs", [{"coverage": 1.5}, {"coverage": 0.0}, {"gamma": 0}])
def test_gate_rejects_bad_config(kwargs) -> None:
    rng = np.random.default_rng(9)
    c = np.abs(rng.standard_normal((8, 16)))
    p = _softmax(rng.standard_normal((8, 40)))
    assert not gss.GSSFeasibilityGate(**kwargs).evaluate(c, p, p.copy()).go

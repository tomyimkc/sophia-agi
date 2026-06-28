# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the Recurrent-Depth Transformer (RDT) nano study.

Pure-stdlib (no numpy/torch), so this runs everywhere the rest of the nano
pretraining suite runs. The invariants check the three mechanisms the OpenMythos
looped-transformer thesis rests on against closed-form controls — a finite-difference
gradient check, the exact diagonal spectral radius, the 1/V chance floor, and exact
parameter counts.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.architecture import recurrent_depth as rd  # noqa: E402


def test_offline_invariants() -> None:
    ok, detail = rd.offline_invariants()
    assert ok, detail["checks"]


def test_bptt_gradient_matches_finite_difference() -> None:
    """The hand-written backprop-through-time must match numerical gradients —
    this is what makes every measured loss in the study real, not asserted."""
    ok, max_rel = rd._numeric_grad_check(seed=0)
    assert ok, f"max relative grad error {max_rel} exceeds 1e-4"


def test_constrained_spectral_radius_below_one_by_construction() -> None:
    """sigmoid(theta) ∈ (0,1) ⇒ the diagonal LTI spectral radius is < 1 for ANY
    weights — the structural stability guarantee, not an empirical observation."""
    import random
    rng = random.Random(0)
    for _ in range(20):
        m = rd.NanoRDT(6, 8, 4, constrained=True, seed=rng.randrange(1 << 30),
                       a_init=rng.uniform(-3, 6))
        assert m.diag_spectral_radius(0) < 1.0


def test_unconstrained_can_exceed_one() -> None:
    """The ablation (A = theta, free) is allowed to exceed 1 — that is the
    instability the constraint removes."""
    m = rd.NanoRDT(6, 8, 4, constrained=False, seed=0, a_init=3.0)
    assert m.diag_spectral_radius(0) >= 1.0


def test_constrained_state_stays_bounded_over_depth() -> None:
    """Forward free-run: constrained ‖h_K‖ must stay finite and below the
    unconstrained norm at the deepest K (the contraction effect)."""
    rep = rd.run_study(quick=True, seed=0)
    st = rep["stability"]
    assert st["constrained_state_more_bounded"] is True
    assert st["unconstrained_grows_faster"] is True


def test_depth_extrapolation_beats_chance() -> None:
    """A weight-shared RDT trained on shallow hops, evaluated at deeper unseen
    hops by running more loops, must beat the 1/V chance floor."""
    rep = rd.run_study(quick=True, seed=0)
    x = rep["depth_extrapolation"]
    assert x["extrapolates_above_chance"] is True
    assert x["mean_extrapolation_acc"] > x["chance"]


def test_weight_sharing_saves_block_params() -> None:
    """The shared RDT reuses one block; the unshared net pays depth× the block
    params for the same forward compute."""
    rep = rd.run_study(quick=True, seed=0)
    pe = rep["parameter_efficiency"]
    assert pe["shared"]["block_params"] < pe["unshared"]["block_params"]
    assert pe["unshared_block_param_multiple"] > 1.0


def test_permutation_iteration_is_correct() -> None:
    """The task oracle (apply π n times) is a genuine bijection power — sanity
    check the supervision targets the model is graded against."""
    perm = rd.make_permutation(7, seed=3)
    assert sorted(perm) == list(range(7))
    for s in range(7):
        assert rd.pi_pow(perm, s, 0) == s
        assert rd.pi_pow(perm, s, 1) == perm[s]
        assert rd.pi_pow(perm, s, 2) == perm[perm[s]]


def test_deterministic_across_runs() -> None:
    """Same seed → identical report (pure-Python, fully seeded)."""
    a = rd.run_study(quick=True, seed=0)
    b = rd.run_study(quick=True, seed=0)
    assert a == b


def test_scope_present() -> None:
    """The honest-scope caveat carries the load-bearing phrase deferring any
    'better than X' claim to the no-overclaim gate."""
    rep = rd.run_study(quick=True, seed=0)
    assert rd.SCOPE_KEY.lower() in rep["honest_scope"].lower()

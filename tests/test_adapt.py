# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for adaptive mixed-precision quantization (numpy, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from moe import adapt  # noqa: E402


def test_adapt_offline_invariants() -> None:
    ok, detail = adapt.offline_invariants()
    assert ok, detail["checks"]


def test_bit_allocator_respects_budget() -> None:
    # Target must be feasible: protected floors (6-bit) + min (1-bit) seed an average
    # above 2.0, so use 4.0 which is comfortably feasible.
    profiles = [
        ("embed", 1000, 8.0, True),
        ("router", 100, 9.0, True),
        ("ffn", 2000, 1.0, False),
        ("expert", 2000, 0.5, False),
    ]
    target = 4.0
    bits = adapt.bit_allocator(profiles, target)
    budget = target * sum(p[1] for p in profiles)
    used = adapt.total_bits(bits, profiles)
    assert used <= budget + 1e-6


def test_protected_floor_never_violated() -> None:
    profiles = [("embed", 1000, 8.0, True), ("ffn", 2000, 1.0, False)]
    for target in (3.5, 4.0, 6.0):  # feasible targets at/above the seeded average
        bits = adapt.bit_allocator(profiles, target)
        assert bits["embed"] >= adapt.PROTECTED_FLOOR


def test_infeasible_target_raises() -> None:
    profiles = [("w", 100, 1.0, False)]
    with pytest.raises(ValueError):
        adapt.bit_allocator(profiles, 0.5)  # below MIN_BITS


def test_floor_infeasible_target_raises() -> None:
    # A target below the protected-floor-seeded average fails closed, not silently overspends.
    profiles = [("embed", 1000, 8.0, True), ("ffn", 2000, 1.0, False)]
    with pytest.raises(ValueError):
        adapt.bit_allocator(profiles, 1.5)  # seeding already averages ~2.7 bits


def test_kl_divergence_self_zero() -> None:
    p = np.full((4, 10), 0.1)
    assert abs(adapt.kl_divergence(p, p)) < 1e-12


def test_one_bit_quantize_is_sign_mean() -> None:
    rng = np.random.default_rng(0)
    W = rng.standard_normal((32, 32))
    q = adapt.quantize_uniform(W, 1)
    assert np.allclose(np.unique(np.abs(q)), np.mean(np.abs(W)))


def test_sensitivity_aware_allocation() -> None:
    # A high-sensitivity non-protected tensor should get >= a low-sensitivity one.
    profiles = [("hot", 100, 9.0, False), ("cold", 100, 0.1, False)]
    bits = adapt.bit_allocator(profiles, 4.0)
    assert bits["hot"] >= bits["cold"]

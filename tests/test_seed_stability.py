# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Seed-stability power audit: distinguishes real seed variance from small-N noise."""

from __future__ import annotations

from tools.seed_stability import analyze, analyze_key, binom_se, required_n


def test_binom_se_known():
    assert abs(binom_se(0.5, 100) - 0.05) < 1e-9


def test_required_n_for_5pp():
    # ~385 samples to resolve +/-5pp at p=0.5, 95% confidence
    assert required_n(0.5, 0.05) == 385


def test_identical_seeds_within_noise():
    v = analyze_key([(22, 32), (22, 32), (22, 32)])
    assert v["observedSeedStdev"] == 0.0
    assert v["withinSamplingNoise"] is True


def test_tiny_eval_swing_is_within_noise():
    # religion 1/6, 0/6, 1/6 -> spread is within binomial noise at N=6
    v = analyze_key([(1, 6), (0, 6), (1, 6)])
    assert v["withinSamplingNoise"] is True
    assert v["requiredNForPlusMinus5pp"] > 100


def test_large_real_gap_exceeds_noise():
    # a big, consistent spread at large N is flagged as possibly real
    v = analyze_key([(900, 1000), (500, 1000), (700, 1000)])
    assert v["withinSamplingNoise"] is False


def test_analyze_flags_underpowered_total():
    seeds = [
        {"total": [22, 32], "religion": [1, 6]},
        {"total": [20, 32], "religion": [0, 6]},
        {"total": [21, 32], "religion": [1, 6]},
    ]
    rep = analyze(seeds)
    assert "total" in rep["underpoweredKeys"]
    assert "religion" in rep["underpoweredKeys"]

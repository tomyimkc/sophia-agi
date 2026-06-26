#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the 'deliberation has a roofline' thesis.

Pure stdlib, deterministic (seeded), offline. The strong check is that the Monte-Carlo
simulation reproduces the closed-form prediction — if the model were wrong, they'd diverge.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.deliberation_roofline import (  # noqa: E402
    DEFAULT_N_LIST,
    DIFFICULTY_PROFILE,
    SCENARIOS,
    ceiling,
    closed_form_curve,
    main,
    monte_carlo_curve,
    run_experiment,
)


def test_oracle_ceiling_is_one():
    assert ceiling(DIFFICULTY_PROFILE, SCENARIOS["oracle"]) > 0.999


def test_leaky_verifier_caps_ceiling_below_one():
    cap = ceiling(DIFFICULTY_PROFILE, SCENARIOS["leaky"])
    assert 0.0 < cap < 0.85  # H3: a leaky verifier hard-caps quality regardless of compute


def test_coverage_rises_monotonically_to_one():
    curve = closed_form_curve(DIFFICULTY_PROFILE, SCENARIOS["good"], DEFAULT_N_LIST)
    covs = [c.coverage for c in curve]
    assert all(covs[i] <= covs[i + 1] + 1e-12 for i in range(len(covs) - 1))
    assert covs[-1] > 0.99


def test_quality_curve_is_concave():  # H1: diminishing returns
    curve = closed_form_curve(DIFFICULTY_PROFILE, SCENARIOS["leaky"], DEFAULT_N_LIST)
    q = [c.quality for c in curve]
    gains = [q[i + 1] - q[i] for i in range(len(q) - 1)]
    assert all(gains[i + 1] <= gains[i] + 1e-9 for i in range(len(gains) - 1)), gains


def test_monte_carlo_matches_closed_form():  # the model is sound, not asserted
    for v in SCENARIOS.values():
        thy = closed_form_curve(DIFFICULTY_PROFILE, v, DEFAULT_N_LIST)
        mc = monte_carlo_curve(DIFFICULTY_PROFILE, v, DEFAULT_N_LIST, trials=400, seed=99)
        max_err = max(abs(m.quality - t.quality) for m, t in zip(mc, thy))
        assert max_err < 0.02, (v.name, max_err)


def test_finite_ridge_point_exists():  # H2
    res = run_experiment(trials=400, seed=7)
    for s in SCENARIOS:
        assert res["scenarios"][s]["verdict"]["ridge_n"] < DEFAULT_N_LIST[-1]


def test_determinism_same_seed_same_result():
    a = run_experiment(trials=300, seed=42)
    b = run_experiment(trials=300, seed=42)
    qa = [pt["quality"] for pt in a["scenarios"]["leaky"]["mc"]]
    qb = [pt["quality"] for pt in b["scenarios"]["leaky"]["mc"]]
    assert qa == qb


def test_cli_self_test_and_run():
    assert main(["--self-test"]) == 0
    assert main(["--run", "--trials", "200"]) == 0
    assert main(["--json", "--trials", "200"]) == 0

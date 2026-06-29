#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the reflex ROC/SNR harness (instinct system, measurement half).

Pure stdlib, deterministic, offline. Central checks: the harness (R1) separates errored
from clean items with the right sign, (R2) reports a finite positive d′, (R3) collapses
to chance on a no-signal control (it must not manufacture separation), (R4) tracks the
sampler's competence, and (R5) computes a stable break-even bar to test the reflex against.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_reflex_eval import (  # noqa: E402
    auc,
    breakeven_snr,
    competence_sweep,
    d_prime,
    load_cases,
    main,
    noise_sampler,
    run_reflex_eval,
)


def test_self_test_entrypoint():
    assert main(["--self-test"]) == 0


def test_dataset_loads():
    cases = load_cases()
    assert len(cases) == 50
    assert all("caseType" in c for c in cases)


def test_r1_reflex_separates_errors():
    r = run_reflex_eval(competence=0.62, seed=1234, bar=1.0)
    assert r.auc > 0.65
    assert r.mean_reflex_error > r.mean_reflex_clean


def test_r2_d_prime_finite_positive():
    r = run_reflex_eval(competence=0.62, seed=1234, bar=1.0)
    assert math.isfinite(r.d_prime) and r.d_prime > 0


def test_r3_noise_control_collapses_to_chance():
    nz = run_reflex_eval(competence=0.62, seed=1234, sampler=noise_sampler, bar=1.0)
    assert abs(nz.auc - 0.5) < 0.12
    assert not nz.clears_breakeven


def test_r4_competence_lowers_error():
    sweep = competence_sweep(seed=1234, bar=1.0)
    errs = [row["base_error"] for row in sweep]
    assert errs[0] > errs[-1] + 0.1
    # d′ should be non-decreasing-ish in competence (more competent ⇒ cleaner signal).
    dps = [row["d_prime"] for row in sweep]
    assert dps[-1] > dps[0]


def test_r5_breakeven_bar_is_stable_and_positive():
    bar = breakeven_snr()
    assert 0.0 < bar < float("inf")
    # The reflex only clears the bar at high competence — the honest go/no-go.
    low = run_reflex_eval(competence=0.45, seed=1234, bar=bar)
    high = run_reflex_eval(competence=0.95, seed=1234, bar=bar)
    assert not low.clears_breakeven
    assert high.clears_breakeven


def test_metric_helpers_edge_cases():
    assert auc([], [1.0]) == 0.5
    assert d_prime([], []) == 0.0
    # Perfectly separated, zero variance ⇒ infinite detectability.
    assert d_prime([1.0, 1.0], [0.0, 0.0]) == float("inf")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all instinct_reflex_eval tests passed")

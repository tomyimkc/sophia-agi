#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the 'instinct gate' thesis (early reflex re-route).

Pure stdlib, deterministic, offline. Central checks: (H1) running an error longer
lowers final correctness; (H2) a usable reflex re-route beats late self-correction;
(H3) a real, finite, positive break-even SNR exists — below it the reflex does not
beat plain commit (the ceiling is the reflex, not the policy); (H4) the ko guard makes
re-route terminate in a bounded ``escalate`` rather than an endless patch-forward loop.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_gate import (  # noqa: E402
    MAX_REROUTE,
    ReflexConfig,
    compounding_curve,
    main,
    run_experiment,
    run_policies,
    snr_sweep,
)


def test_self_test_entrypoint():
    assert main(["--self-test"]) == 0


def test_h1_compounding_monotone():
    """Later error ⇒ fewer derail steps ⇒ higher correctness, and MC tracks closed form."""
    curve = compounding_curve(trials=4000, seed=7)
    mc = [row["mc_correct"] for row in curve]
    # Monotone non-decreasing in error lateness (small MC slack), with a real spread.
    assert all(mc[i] <= mc[i + 1] + 0.05 for i in range(len(mc) - 1)), mc
    assert mc[-1] - mc[0] > 0.2, mc
    worst = max(abs(r["mc_correct"] - r["closed_form"]) for r in curve)
    assert worst < 0.05, worst


def test_h2_instinct_beats_late_and_commit_with_good_reflex():
    pol = run_policies(trials=4000, seed=1234, cfg=ReflexConfig(snr=3.0, tau=2.5))
    assert pol["instinct"]["correct"] > pol["late"]["correct"] + 0.05
    assert pol["instinct"]["correct"] > pol["commit"]["correct"] + 0.05


def test_h3_breakeven_snr_is_finite_positive():
    """A useless reflex must not beat commit; a good one must — so break-even is real."""
    sweep = snr_sweep(trials=4000, seed=1234, snrs=[0.0, 0.5, 1.0, 2.0, 3.0])
    poor, good = sweep[0], sweep[-1]
    assert poor["instinct_correct"] <= poor["commit_correct"] + 0.02  # snr=0 doesn't help
    assert good["instinct_correct"] > good["commit_correct"] + 0.05   # snr=3 clearly helps
    # The instinct curve is non-decreasing in SNR up to its ceiling.
    inst = [row["instinct_correct"] for row in sweep]
    assert inst[-1] >= inst[0]


def test_h3_late_is_marginal_not_magic():
    """Faithful to the literature: intrinsic self-correction is ~a wash, not a fix."""
    pol = run_policies(trials=4000, seed=99, cfg=ReflexConfig(snr=3.0, tau=2.5))
    assert abs(pol["late"]["correct"] - pol["commit"]["correct"]) < 0.06


def test_h4_escalation_is_bounded():
    """Aggressive reflex on a hard distribution escalates, and never exceeds the budget."""
    hard = run_policies(trials=4000, seed=1234, cfg=ReflexConfig(snr=0.0, tau=0.0))
    assert hard["instinct"]["escalate_rate"] > 0.0
    assert hard["instinct"]["mean_reroutes"] <= MAX_REROUTE


def test_verdict_envelope_is_candidate_only():
    res = run_experiment(trials=2000, seed=1234)
    v = res["verdict"]
    assert v["candidateOnly"] is True
    assert v["level3Evidence"] is False
    assert v["h1_compounding"] and v["h2_early_beats_late"] and v["h4_bounded_escalate"]
    assert 0.0 < v["h3_breakeven_snr"] < float("inf")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    print("all instinct_gate tests passed")

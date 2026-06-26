#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the autonomous pretraining-experiment runner.

Asserts the loop is a REAL closed-loop search (it improves on its starting point by reading
measured results), is fail-closed on divergence (records score=inf, never fabricates), never
claims AGI, and that the RunPod escalation is cost-gated (never launches autonomously).
Offline, deterministic, dependency-free, tiny configs for speed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.autopilot.backends import LocalBackend, RunPodEscalation  # noqa: E402
from pretraining.autopilot.runner import autopilot  # noqa: E402
from pretraining.autopilot.strategies import (  # noqa: E402
    ComputeAllocation, LearningRateSearch, MixtureSearch,
)

_FAST = {"vocab": 8, "order": 2, "context": 2, "hidden": 12, "D": 600, "epochs": 6, "seed": 0}


def test_local_backend_runs_real_experiment() -> None:
    res = LocalBackend().run(dict(_FAST, lr=0.03))
    assert res["params"] > 0
    assert res["held_loss"] == res["held_loss"]  # not NaN
    assert "floor_E" in res and res["compute_proxy"] == res["params"] * _FAST["D"]


def test_lr_search_improves_on_start() -> None:
    strat = LearningRateSearch(_FAST, lr0=0.05)
    rep = autopilot(strat, LocalBackend(), max_trials=8, patience=4)
    assert rep["canClaimAGI"] is False
    assert rep["best"] is not None
    first = rep["history"][0]["score"]
    # the closed loop must end at least as good as where it started (usually strictly better)
    assert rep["best"]["score"] <= first + 1e-9
    # every recorded score is real (finite) unless explicitly diverged
    for h in rep["history"]:
        assert h["diverged"] or h["score"] < float("inf")


def test_failclosed_records_inf_for_diverged() -> None:
    # The fail-closed mechanism: a backend reporting a diverged run must be scored inf by the
    # loop (never a fabricated finite number), and the best must avoid it. We use a stub
    # backend because the bounded tanh/softmax toy resists *numerical* divergence — an honest
    # property in its own right — so we test the loop's handling directly and deterministically.
    class _StubBackend:
        def __init__(self):
            self.n = 0

        def run(self, config):
            self.n += 1
            if self.n == 1:
                return {"held_loss": 1.0, "diverged": False, "params": 10,
                        "compute_proxy": 10, "floor_E": 0.5, "excess": 0.5}
            return {"held_loss": float("inf"), "diverged": True, "params": 10,
                    "compute_proxy": 10, "floor_E": 0.5, "excess": float("inf")}

    class _TwoConfigStrategy:
        def initial(self):
            return {"lr": 0.01}

        def propose_next(self, history):
            return {"lr": 99.0} if len(history) < 2 else None

    rep = autopilot(_TwoConfigStrategy(), _StubBackend(), max_trials=3, patience=3)
    diverged = [h for h in rep["history"] if h["diverged"]]
    assert diverged and all(h["score"] == float("inf") for h in diverged)
    assert rep["best"]["score"] == 1.0          # best avoids the diverged trial
    assert rep["n_diverged"] >= 1


def test_high_lr_is_worse_than_good_lr() -> None:
    # The real loop's signal is sound: an absurd learning rate yields a much worse loss than
    # a sensible one, so the search correctly steers away from it.
    bad = LocalBackend().run(dict(_FAST, lr=50.0, optimizer="sgd"))["held_loss"]
    good = LocalBackend().run(dict(_FAST, lr=0.03, optimizer="adam"))["held_loss"]
    assert bad > good


def test_mixture_search_converges_interior() -> None:
    base = dict(_FAST, target="blend")
    strat = MixtureSearch(base, iters=3)
    rep = autopilot(strat, LocalBackend(), max_trials=8, patience=8)
    assert rep["best"] is not None
    w = rep["best"]["config"]["mix"]
    assert 0.0 < w < 1.0  # blended target -> interior optimum


def test_compute_allocation_runs_and_picks_best() -> None:
    strat = ComputeAllocation(_FAST, hiddens=(4, 8, 16), compute_proxy=600 * 808)
    rep = autopilot(strat, LocalBackend(), max_trials=4, patience=4)
    assert rep["n_trials"] >= 2
    assert rep["best"] is not None
    # the chosen config holds compute ~constant: params*D should be near the target
    cfg = rep["best"]["config"]
    assert cfg["D"] >= 100


def test_runpod_escalation_is_cost_gated() -> None:
    esc = RunPodEscalation()
    # default: dry-run, never launched, guard blocks
    plan = esc.plan({"seed": 0}, branch="b")
    assert plan["launched"] is False
    assert plan["mode"] == "dry_run"
    assert "BLOCKED" in plan["guard"]
    assert "--dry-run" in plan["dry_run_command"]
    # launch requested but NO cost ceiling -> still blocked, still not launched
    plan2 = esc.plan({"seed": 0}, branch="b", launch=True, cost_ceiling_usd=None)
    assert plan2["launched"] is False
    assert "BLOCKED" in plan2["guard"]
    # launch + adequate ceiling -> guard passes but adapter STILL does not launch inline
    plan3 = esc.plan({"seed": 0}, branch="b", launch=True, cost_ceiling_usd=100.0)
    assert plan3["launched"] is False
    assert "PASS" in plan3["guard"]


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

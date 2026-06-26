#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the RunPod calibration / cost-governor machinery (autopilot Step 1).

Asserts the safety invariants that make autonomous GPU search OK: the cost governor is
fail-closed against a hard ceiling, the eval-ladder objective parser is faithful + fails
closed, the backend never launches inline (always dry-run command unless explicitly built
for execute), and the calibration dry-run spends nothing. Offline, deterministic.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.autopilot import calibrate  # noqa: E402
from pretraining.autopilot.cost_governor import BudgetExceeded, CostGovernor  # noqa: E402
from pretraining.autopilot.eval_ladder_objective import parse_objective  # noqa: E402
from pretraining.autopilot.runpod_backend import RunPodLoRABackend  # noqa: E402


# -- cost governor is fail-closed ----------------------------------------------
def test_governor_blocks_past_ceiling() -> None:
    gov = CostGovernor(1.0, price_per_hr=0.69, est_hours_per_trial=0.75, overhead_frac=0.30)
    assert gov.can_afford(1) is True              # ~$0.67 < $1.00
    assert gov.can_afford(2) is False             # ~$1.34 > $1.00
    assert gov.max_affordable_trials() == 1
    try:
        gov.guard(2)
        assert False, "guard should have raised"
    except BudgetExceeded:
        pass


def test_governor_records_actual_and_refines() -> None:
    gov = CostGovernor(25.0, price_per_hr=0.69)
    ledger = gov.record(0.6, price_per_hr=0.69)
    assert abs(ledger["trial_cost_usd"] - 0.414) < 1e-6
    assert gov.spent() == ledger["trial_cost_usd"]
    assert gov.remaining() == round(25.0 - 0.414, 4)
    # the per-trial estimate is refined from the first measured run
    assert abs(gov.snapshot()["est_hours_per_trial"] - 0.6) < 1e-9


def test_governor_rejects_bad_ceiling() -> None:
    for bad in (0.0, -1.0):
        try:
            CostGovernor(bad)
            assert False, "should reject non-positive ceiling"
        except ValueError:
            pass


# -- objective parser is faithful + fail-closed --------------------------------
def test_objective_parses_real_ladder() -> None:
    sample = ROOT / "training" / "local_sophia_v2" / "eval_ladder_adapter.json"
    if not sample.exists():
        return
    obj = parse_objective(json.loads(sample.read_text(encoding="utf-8")))
    assert obj["ok"] is True
    assert obj["rung_scores"]["base"] is not None
    assert obj["rung_scores"]["adapter+gate"] is not None
    # objective_for_min is the negated uplift -> minimizing it maximizes uplift
    assert obj["objective_for_min"] == round(-obj["uplift_combined"], 4)


def test_objective_fail_closed() -> None:
    assert parse_objective({"schema": "wrong"})["ok"] is False
    assert parse_objective({"schema": "wrong"})["objective_for_min"] == float("inf")
    empty = parse_objective({"schema": "sophia.eval_ladder.v2", "rungs": []})
    assert empty["ok"] is False and empty["objective_for_min"] == float("inf")


# -- backend builds dry-run commands; never launches inline --------------------
def test_backend_command_is_dry_run_by_default() -> None:
    gov = CostGovernor(1.0)
    backend = RunPodLoRABackend(gov, branch="feat/x", model="Qwen/Qwen2.5-3B-Instruct")
    cmd = backend.build_command({"epochs": 1, "seed": 0})
    assert "--dry-run" in cmd and "--yes" not in cmd
    assert "tools/runpod_train.py" in " ".join(cmd)
    # execute variant produces --yes but this never shells out
    ex = backend.build_command({"epochs": 1}, execute=True)
    assert "--yes" in ex and "--dry-run" not in ex
    plan = backend.plan_trial({"epochs": 1})
    assert plan["launched"] is False and plan["affordable"] is True


def test_backend_plan_blocks_when_unaffordable() -> None:
    gov = CostGovernor(0.10)   # below one trial's projected cost
    backend = RunPodLoRABackend(gov, branch="feat/x")
    plan = backend.plan_trial({"epochs": 1})
    assert plan["affordable"] is False and "BLOCKED" in plan["guard"]


# -- calibration dry-run spends nothing ----------------------------------------
def test_calibrate_dry_run_no_spend() -> None:
    rep = calibrate.dry_run("feat/x", "Qwen/Qwen2.5-3B-Instruct", ceiling=1.0, epochs=1)
    assert rep["mode"] == "dry_run"
    assert rep["canClaimAGI"] is False
    assert rep["trial_plan"]["launched"] is False
    assert rep["projected_per_trial_usd"] > 0
    assert len(rep["projected_sweep_tiers"]) == 3


def test_calibrate_from_result_computes_cost() -> None:
    sample = ROOT / "training" / "local_sophia_v2" / "eval_ladder_adapter.json"
    if not sample.exists():
        return
    rep = calibrate.from_result(sample, wall_clock_hours=0.6, price_per_hr=0.69,
                                ceiling=25.0, model="Qwen/Qwen2.5-3B-Instruct")
    assert abs(rep["measured"]["actual_trial_cost_usd"] - 0.414) < 1e-6
    assert rep["uplift_combined"] is not None
    assert rep["sweep_tiers_from_measured_cost"][0]["tier"] == "small"


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

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the ASHA scheduler + LoRA search space (autopilot Step 2, C2/C3).

Asserts successive halving genuinely prunes (executes fewer runs than naive and keeps the
best), is fail-closed on the cost ceiling (never overspends), and that the search space is
honest about which knobs transfer to GPU today. Offline, deterministic, dependency-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.autopilot.asha import successive_halving  # noqa: E402
from pretraining.autopilot.cost_governor import CostGovernor  # noqa: E402
from pretraining.autopilot.search_space import (  # noqa: E402
    NEEDS_PASSTHROUGH, WIRED_TODAY, passthrough_gap, sample_configs,
)


def test_successive_halving_prunes_and_keeps_best() -> None:
    # A deterministic scorer where lower "x" is better and the budget doesn't change ranking.
    configs = [{"x": x} for x in (5, 1, 9, 3, 7, 2)]
    run_fn = lambda cfg, budget: float(cfg["x"])
    rep = successive_halving(configs, run_fn, budgets=[1, 2, 4], eta=2.0)
    assert rep["ok"] is True
    assert rep["best"]["x"] == 1                       # the true best survives
    assert rep["runs_executed"] < rep["naive_runs"]    # cheaper than running all to the top
    assert rep["savings_vs_naive"] > 0


def test_failed_runs_score_inf_and_are_pruned() -> None:
    # Configs with x<0 "diverge" -> inf; they must be pruned first, never chosen.
    configs = [{"x": x} for x in (-1, 2, -3, 1, 4)]
    run_fn = lambda cfg, budget: (float("inf") if cfg["x"] < 0 else float(cfg["x"]))
    rep = successive_halving(configs, run_fn, budgets=[1, 2], eta=2.0)
    assert rep["best"]["x"] == 1
    assert all(c["x"] > 0 for c in rep["rungs"][-1]["promoted"])


def test_cost_governor_truncates_failclosed() -> None:
    configs = [{"x": x} for x in range(8)]
    run_fn = lambda cfg, budget: float(cfg["x"])
    # ceiling only affords a couple of runs; the sweep must not overspend
    gov = CostGovernor(1.0, price_per_hr=0.69, est_hours_per_trial=0.75)  # ~$0.67/run
    rep = successive_halving(configs, run_fn, budgets=[1, 2, 4], eta=2.0, governor=gov)
    assert rep["truncated"] is True
    assert gov.spent() <= gov.ceiling                  # never exceeded the ceiling
    assert rep["cost"]["spent_usd"] <= 1.0


def test_single_budget_runs_once_each() -> None:
    configs = [{"x": x} for x in (3, 1, 2)]
    run_fn = lambda cfg, budget: float(cfg["x"])
    rep = successive_halving(configs, run_fn, budgets=[1], eta=2.0)
    assert rep["runs_executed"] == 3
    assert rep["best"]["x"] == 1


def test_search_space_sampler_is_deterministic_and_distinct() -> None:
    a = sample_configs(8, seed=0)
    b = sample_configs(8, seed=0)
    assert a == b                                       # deterministic
    assert len(a) == 8
    for cfg in a:
        assert cfg["lora_rank"] in (8, 16, 32, 64)
        assert cfg["model"].startswith("Qwen")


def test_passthrough_gap_is_complete() -> None:
    gap = passthrough_gap()
    # the LoRA passthrough is now wired -> the whole space transfers to GPU
    assert "epochs" in WIRED_TODAY and "lr" in WIRED_TODAY and "lora_rank" in WIRED_TODAY
    assert NEEDS_PASSTHROUGH == set()
    assert gap["complete"] is True
    assert set(gap["needs_passthrough"]) == set()


def test_searched_config_transfers_to_runpod_command() -> None:
    # A sampled LoRA config must produce real runpod_train.py override flags (dry-run).
    from pretraining.autopilot.cost_governor import CostGovernor
    from pretraining.autopilot.runpod_backend import RunPodLoRABackend
    cfg = {"lr": 2e-4, "lora_rank": 32, "lora_alpha": 64, "neftune_alpha": 10,
           "epochs": 1, "seed": 0}
    cmd = RunPodLoRABackend(CostGovernor(25.0), branch="b").build_command(cfg)
    s = " ".join(cmd)
    assert "--lr 0.0002" in s and "--lora-r 32" in s and "--lora-alpha 64" in s
    assert "--neftune-alpha 10" in s and "--dry-run" in s


def test_empty_inputs_fail_closed() -> None:
    assert successive_halving([], lambda c, b: 0.0, budgets=[1])["ok"] is False
    assert successive_halving([{"x": 1}], lambda c, b: 0.0, budgets=[])["ok"] is False


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

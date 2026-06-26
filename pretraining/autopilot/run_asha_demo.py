#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Demonstrate ASHA (C3) selecting a good config on REAL measured results — no GPU spend.

Runs successive halving over the local nano backend (real training, free, CPU) to prove the
scheduler prunes bad configs cheaply and surfaces the best one — then reports how the SAME
scheduler would drive the real RunPod LoRA pipeline under the cost governor (projection only,
no pod). Also prints the C2 passthrough gap (which LoRA knobs need a Step-2 change to
transfer to GPU). Writes a JSON report.

    python -m pretraining.autopilot.run_asha_demo --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretraining.autopilot.asha import successive_halving
from pretraining.autopilot.backends import LocalBackend
from pretraining.autopilot.cost_governor import CostGovernor
from pretraining.autopilot.search_space import passthrough_gap, sample_configs

HERE = Path(__file__).resolve().parent


def run(*, quick: bool = False, out: Path | None = None) -> dict:
    base = {"vocab": 8, "order": 2, "context": 2, "hidden": 16, "D": 800, "seed": 0}
    lrs = [0.002, 0.01, 0.03, 0.1, 0.3] + ([] if quick else [1.0])
    configs = [dict(base, lr=lr) for lr in lrs]
    budgets = [1, 3] if quick else [1, 3, 6]

    backend = LocalBackend()

    def run_fn(cfg, budget):
        return backend.run({**cfg, "epochs": int(budget)})["held_loss"]

    nano = successive_halving(configs, run_fn, budgets=budgets, eta=2.0)

    # How the SAME scheduler would behave on real GPU under a ceiling (projection only).
    gov = CostGovernor(25.0)   # a realistic Step-2 ceiling
    affordable_runs = gov.max_affordable_trials()

    report = {
        "study": "ASHA successive-halving demo (real nano results; GPU cost projected)",
        "canClaimAGI": False,
        "honesty_note": ("The selection below is REAL — measured nano losses. The GPU section "
                         "is a projection at the observed $/hr; no pod is created."),
        "nano_selection": {
            "n_configs": nano["n_configs"],
            "runs_executed": nano["runs_executed"],
            "naive_runs": nano["naive_runs"],
            "savings_vs_naive": nano["savings_vs_naive"],
            "best_config": {k: nano["best"][k] for k in ("lr", "hidden", "D")},
            "rungs": [{"rung": r["rung"], "budget": r["budget"], "n_in": r["n_in"],
                       "kept": r["n_kept"], "best_score": r["best_score"]}
                      for r in nano["rungs"]],
        },
        "gpu_projection": {
            "ceiling_usd": gov.ceiling,
            "est_cost_per_trial_usd": gov.estimate_trial(),
            "max_affordable_trials": affordable_runs,
            "note": ("at the observed $0.69/hr, a $25 ceiling funds ~%d real trials — enough "
                     "for an ASHA sweep over ~10-16 configs with promotions" % affordable_runs),
        },
        "c2_passthrough_gap": passthrough_gap(),
    }
    out = out or (HERE / "asha-demo-latest.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    r = run(quick=args.quick, out=args.out)
    s = r["nano_selection"]
    print(f"ASHA over {s['n_configs']} configs: {s['runs_executed']} runs "
          f"(naive {s['naive_runs']}, saved {s['savings_vs_naive']*100:.0f}%)")
    for rung in s["rungs"]:
        print(f"  rung {rung['rung']} budget={rung['budget']} "
              f"{rung['n_in']}->{rung['kept']} best={rung['best_score']}")
    print(f"BEST: {s['best_config']}")
    g = r["gpu_projection"]
    print(f"GPU projection: ${g['est_cost_per_trial_usd']}/trial, "
          f"${g['ceiling_usd']} ceiling -> ~{g['max_affordable_trials']} real trials")


if __name__ == "__main__":
    main()

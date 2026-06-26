#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibration harness — replace every cost ESTIMATE with one MEASURED number.

Step 1 of the real-pipeline plan (pretraining/autopilot/SCOPE.md): run exactly ONE real
RunPod LoRA trial, measure its true wall-clock and cost, and extrapolate honest budgets for
the Step-2 sweep. Two modes, and neither one spends money by itself:

  --dry-run (default)
      Build the runpod_train.py command, project the cost against the ceiling, and write a
      plan. No pod, no cost. This is what CI and tests exercise.

  --from-result <eval_ladder_adapter.json> --wall-clock-hours H [--price-per-hr P]
      POST-PROCESS a completed real run (the pod was launched by calibrate-runpod.yml via
      runpod_train.py --yes). Computes ACTUAL cost = H × P, parses the objective uplift, and
      re-projects the sweep tiers from the measured per-trial cost.

The actual paid launch lives in the GitHub Actions workflow, gated behind an explicit
ceiling + a typed confirmation — so spending is always a deliberate human action.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretraining.autopilot.cost_governor import CostGovernor, OBSERVED_PRICE_PER_HR
from pretraining.autopilot.runpod_backend import DEFAULT_MODEL, RunPodLoRABackend

HERE = Path(__file__).resolve().parent

# Sweep tiers from SCOPE.md, as (label, effective_full_trials) — what Step 2 would cost
# once the per-trial cost is measured here.
SWEEP_TIERS = [("small", 10), ("medium", 24), ("generous", 40)]


def _project_tiers(cost_per_trial: float) -> list[dict]:
    out = []
    for label, n in SWEEP_TIERS:
        out.append({"tier": label, "trials": n,
                    "est_cost_usd": round(cost_per_trial * n, 2)})
    return out


def dry_run(branch: str, model: str, ceiling: float, epochs: int) -> dict:
    gov = CostGovernor(ceiling)
    backend = RunPodLoRABackend(gov, branch=branch, model=model)
    config = {"model": model, "epochs": epochs, "seed": 0}
    plan = backend.plan_trial(config)
    return {
        "mode": "dry_run",
        "canClaimAGI": False,
        "honesty_note": ("No pod, no cost. Projections use the observed $/hr anchor; the real "
                         "per-trial cost is unknown until a --from-result calibration run."),
        "model": model,
        "branch": branch,
        "ceiling_usd": ceiling,
        "trial_plan": plan,
        "projected_per_trial_usd": gov.estimate_trial(),
        "projected_sweep_tiers": _project_tiers(gov.estimate_trial()),
        "next_step": ("trigger .github/workflows/calibrate-runpod.yml with a cost_ceiling and "
                      "confirm=SPEND to run ONE real trial, then re-run this with --from-result"),
    }


def from_result(result_path: Path, wall_clock_hours: float, price_per_hr: float,
                ceiling: float, model: str) -> dict:
    report = json.loads(result_path.read_text(encoding="utf-8"))
    gov = CostGovernor(ceiling, price_per_hr=price_per_hr)
    backend = RunPodLoRABackend(gov, branch="(measured)", model=model)
    ledger = gov.record(wall_clock_hours, price_per_hr=price_per_hr)
    objective = backend.score_result(report)
    cost_per_trial = ledger["trial_cost_usd"]
    return {
        "mode": "from_result",
        "canClaimAGI": False,
        "measured": {
            "wall_clock_hours": wall_clock_hours,
            "price_per_hr": price_per_hr,
            "actual_trial_cost_usd": cost_per_trial,
        },
        "objective": objective,
        "uplift_combined": objective.get("uplift_combined"),
        "cost_ledger": gov.snapshot(),
        "sweep_tiers_from_measured_cost": _project_tiers(cost_per_trial),
        "verdict": ("calibration complete — use these MEASURED tier costs to pick a ceiling "
                    "for the Step-2 sweep"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--branch", default="claude/deepseek-pretraining-alignment-o281ju")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--ceiling", type=float, default=1.0, help="hard USD ceiling for the trial")
    ap.add_argument("--from-result", type=Path, default=None,
                    help="post-process a completed eval_ladder_adapter.json")
    ap.add_argument("--wall-clock-hours", type=float, default=None)
    ap.add_argument("--price-per-hr", type=float, default=OBSERVED_PRICE_PER_HR)
    ap.add_argument("--out", type=Path, default=HERE / "calibration-latest.json")
    args = ap.parse_args()

    if args.from_result:
        if args.wall_clock_hours is None:
            raise SystemExit("--from-result requires --wall-clock-hours")
        report = from_result(args.from_result, args.wall_clock_hours, args.price_per_hr,
                             args.ceiling, args.model)
    else:
        report = dry_run(args.branch, args.model, args.ceiling, args.epochs)

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

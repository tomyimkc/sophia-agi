#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the autonomous pretraining-experiment runner end to end.

    python -m pretraining.autopilot.run_autopilot --task lr        # tune learning rate
    python -m pretraining.autopilot.run_autopilot --task mixture   # search data 配比
    python -m pretraining.autopilot.run_autopilot --task compute   # compute-optimal N vs D
    python -m pretraining.autopilot.run_autopilot --task lr --escalate --branch <b>

The loop runs REAL nano experiments locally (free, CPU). ``--escalate`` additionally prints
a GATED RunPod plan for the winning config — DRY-RUN by default; it never spends GPU money
without an explicit ``--launch`` + ``--cost-ceiling`` + RUNPOD_API_KEY (and even then it only
emits the command for you to run, by design). Writes a JSON report next to the script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretraining.autopilot.backends import LocalBackend, RunPodEscalation
from pretraining.autopilot.runner import autopilot
from pretraining.autopilot.strategies import (
    ComputeAllocation, LearningRateSearch, MixtureSearch,
)

HERE = Path(__file__).resolve().parent


def _build(task: str, quick: bool):
    base = {"vocab": 8, "order": 2, "context": 2, "hidden": 16, "D": 1200,
            "epochs": 8 if quick else 12, "seed": 0}
    if task == "lr":
        return LearningRateSearch(base, lr0=0.05), 10
    if task == "mixture":
        b = dict(base); b["target"] = "blend"
        return MixtureSearch(b, iters=3 if quick else 4), 10
    if task == "compute":
        hiddens = (4, 8, 16, 32) if quick else (4, 8, 16, 32, 64)
        return ComputeAllocation(base, hiddens=hiddens), len(hiddens)
    raise SystemExit(f"unknown task: {task}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["lr", "mixture", "compute"], default="lr")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--escalate", action="store_true", help="print a gated RunPod plan for the winner")
    ap.add_argument("--branch", default="claude/deepseek-pretraining-alignment-o281ju")
    ap.add_argument("--launch", action="store_true", help="REQUEST a real GPU run (still requires ceiling+key)")
    ap.add_argument("--cost-ceiling", type=float, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    strategy, max_trials = _build(args.task, args.quick)
    report = autopilot(strategy, LocalBackend(), max_trials=max_trials)
    report["task"] = args.task

    if args.escalate and report["best"]:
        report["runpod_escalation"] = RunPodEscalation().plan(
            report["best"]["config"], branch=args.branch, launch=args.launch,
            cost_ceiling_usd=args.cost_ceiling)

    out = args.out or (HERE / f"autopilot-{args.task}-latest.json")
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"task={args.task}  trials={report['n_trials']}  diverged={report['n_diverged']}  "
          f"stop={report['stop_reason']}")
    for h in report["history"]:
        mark = "*" if h["is_best_so_far"] else " "
        print(f" {mark} trial {h['trial']:2d}  score={h['score']}  "
              f"{ {k: h['config'][k] for k in ('lr','hidden','D','mix') if k in h['config']} }")
    if report["best"]:
        b = report["best"]
        print(f"BEST: score={b['score']}  config="
              f"{ {k: b['config'][k] for k in ('lr','hidden','D','mix') if k in b['config']} }")
    if report.get("runpod_escalation"):
        esc = report["runpod_escalation"]
        print(f"\nRunPod escalation [{esc['mode']}] {esc['guard']}")
        print(f"  dry-run: {esc['dry_run_command']}")
        print(f"  est cost: {esc['est_cost_usd']['range']} USD on {esc['est_cost_usd']['gpu']}")


if __name__ == "__main__":
    main()

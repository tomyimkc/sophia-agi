#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sweep checkpoint cadence against a node-failure rate and emit a goodput report.

The classic large-scale-training resilience question: given an MTBF, what checkpoint
interval maximizes goodput (useful work that survives) net of wasted compute and
recovery overhead? Too-frequent checkpoints waste I/O; too-rare ones waste re-run.
This sweeps `--checkpoints` on one seeded fault trace and reports the trade-off.

    python tools/run_cluster_faultsim.py                  # default sweep
    python tools/run_cluster_faultsim.py --mtbf 600 --checkpoints 60,180,300,600,1200 \
        --jobs 120 --seed 3 --markdown
    python tools/run_cluster_faultsim.py --out agi-proof/benchmark-results/cluster/faults.public-report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clustersim.faults import inject_faults, simulate_with_faults
from clustersim.job import synthetic_trace
from clustersim.scheduler import get_policy
from clustersim.topology import homogeneous_cluster


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--islands", type=int, default=2)
    p.add_argument("--jobs", type=int, default=120)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=float, default=3600.0)
    p.add_argument("--policy", default="backfill-topo")
    p.add_argument("--mtbf", type=float, default=600.0, help="mean time between node failures (s)")
    p.add_argument("--recovery", type=float, default=60.0, help="reschedule+reload latency (s)")
    p.add_argument("--checkpoints", default="60,180,300,600,1200",
                   help="comma list of checkpoint intervals (s) to sweep")
    p.add_argument("--out", default=None)
    p.add_argument("--markdown", action="store_true")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    base_cluster = homogeneous_cluster(
        nodes=args.nodes, gpus_per_node=args.gpus_per_node, islands_per_node=args.islands)
    base_trace = synthetic_trace(n_jobs=args.jobs, seed=args.seed, horizon_s=args.horizon)
    node_ids = [n.id for n in base_cluster.nodes]
    faults = inject_faults(node_ids, seed=args.seed, mtbf_s=args.mtbf, horizon_s=args.horizon)
    intervals = [float(x) for x in args.checkpoints.split(",") if x.strip()]

    sweep = []
    for ckpt in intervals:
        cl = deepcopy(base_cluster)
        tr = deepcopy(base_trace)
        res = simulate_with_faults(
            cl, tr, get_policy(args.policy), list(faults),
            checkpoint_s=ckpt, recovery_s=args.recovery)
        sweep.append(res.as_dict())

    best = max(sweep, key=lambda r: r["goodput"]) if sweep else None
    return {
        "schema": "sophia.cluster_fault_sim.v1",
        "scope": "SIMULATED — illustrative MTBF/checkpoint constants; not a real-fleet measurement.",
        "config": {
            "nodes": args.nodes, "gpus_per_node": args.gpus_per_node,
            "total_gpus": base_cluster.total_gpus, "jobs": args.jobs, "seed": args.seed,
            "policy": args.policy, "mtbf_s": args.mtbf, "recovery_s": args.recovery,
            "n_faults": len(faults), "horizon_s": args.horizon,
        },
        "best_checkpoint_s": best["checkpoint_s"] if best else None,
        "sweep": sweep,
    }


def _markdown(report: dict) -> str:
    cols = ["checkpoint_s", "goodput", "raw_utilization", "wasted_fraction",
            "total_restarts", "recovery_p99_s", "completed"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in report["sweep"]]
    note = f"\nbest goodput at checkpoint_s = {report['best_checkpoint_s']}"
    return "\n".join([head, sep, *rows]) + note


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(args)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    if args.markdown or not args.out:
        print(_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

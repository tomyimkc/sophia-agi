#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the cluster scheduler simulator across policies and emit a measured trade-off report.

Replays one seeded job trace on one cluster under every placement policy, so the numbers
are directly comparable, and writes a *.public-report.json with the utilization /
queue-latency / fragmentation trade-off the JD calls out. Deterministic given the seed.

    python tools/run_cluster_sim.py                       # default 8x8 cluster, 200 jobs
    python tools/run_cluster_sim.py --nodes 16 --gpus-per-node 8 --islands 2 \
        --jobs 400 --seed 7 --markdown
    python tools/run_cluster_sim.py --out agi-proof/benchmark-results/cluster/scheduler.public-report.json
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

from cluster.job import synthetic_trace
from cluster.scheduler import POLICIES, get_policy
from cluster.simulator import simulate
from cluster.topology import homogeneous_cluster


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--islands", type=int, default=2, help="NVLink islands per node")
    p.add_argument("--gpus-per-rack", type=int, default=0, help="0 = single rack")
    p.add_argument("--jobs", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=float, default=3600.0, help="arrival window (s)")
    p.add_argument("--island-tax", type=float, default=0.06)
    p.add_argument("--node-tax", type=float, default=0.12)
    p.add_argument("--policies", default=",".join(POLICIES), help="comma list")
    p.add_argument("--out", default=None, help="write JSON report here")
    p.add_argument("--markdown", action="store_true", help="also print a markdown table")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    base_cluster = homogeneous_cluster(
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        islands_per_node=args.islands,
        gpus_per_rack=args.gpus_per_rack,
    )
    base_trace = synthetic_trace(n_jobs=args.jobs, seed=args.seed, horizon_s=args.horizon)
    names = [n.strip() for n in args.policies.split(",") if n.strip()]

    results = []
    for name in names:
        # Fresh cluster + trace per policy so they are independent and comparable.
        cl = deepcopy(base_cluster)
        tr = deepcopy(base_trace)
        res = simulate(cl, tr, get_policy(name),
                       island_tax=args.island_tax, node_tax=args.node_tax)
        results.append(res.as_dict())

    return {
        "schema": "sophia.cluster_scheduler_sim.v1",
        "scope": "SIMULATED — illustrative network-tax constants; not a real-fleet measurement.",
        "config": {
            "nodes": args.nodes, "gpus_per_node": args.gpus_per_node,
            "total_gpus": base_cluster.total_gpus, "islands_per_node": args.islands,
            "gpus_per_rack": args.gpus_per_rack, "jobs": args.jobs, "seed": args.seed,
            "horizon_s": args.horizon, "island_tax": args.island_tax, "node_tax": args.node_tax,
        },
        "results": results,
    }


def _markdown(report: dict) -> str:
    cols = ["policy", "utilization", "throughput_jobs_per_hr", "wait_p50_s",
            "wait_p99_s", "mean_fragmentation", "mean_network_tax"]
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in report["results"]]
    return "\n".join([head, sep, *rows])


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

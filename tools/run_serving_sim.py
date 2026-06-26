#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Inference-serving M0 — static vs continuous batching simulator (offline).

Deterministic, no GPU. Runs a seeded Poisson workload through both scheduling
policies and reports TTFT / TPOT / throughput / goodput, plus the continuous-over-
static uplift — the CI-green floor of the serving track. The real engine (model
driven through the Rust paged prefix KV-cache) is the GPU milestone.

    python tools/run_serving_sim.py --mode mock

Honest scope: a SIMULATION (like clustersim/), not a served model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.sim.engine import compare, simulate  # noqa: E402
from serving.sim.workload import poisson_workload  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "serving" / "serving-sim.public-report.json"


def _offline_invariants(seed: int = 0) -> "tuple[bool, dict]":
    reqs = poisson_workload(24, rate=0.8, seed=seed)
    cmp = compare(reqs, max_batch=4, prefill_chunk=16, ttft_slo=40, tpot_slo=2.0)
    s = simulate(reqs, policy="static", max_batch=4)
    c = simulate(reqs, policy="continuous", max_batch=4)
    checks = {
        "allFinishedStatic": s.n == 24,
        "allFinishedContinuous": c.n == 24,
        "continuousThroughputGE": c.throughput >= s.throughput,
        "continuousTtftLE": c.ttft_mean <= s.ttft_mean,
        "goodputBounded": 0.0 <= c.goodput <= 1.0 and 0.0 <= s.goodput <= 1.0,
        "makespanPositive": c.makespan > 0 and s.makespan > 0,
        "deterministic": compare(poisson_workload(24, rate=0.8, seed=seed),
                                 max_batch=4, prefill_chunk=16, ttft_slo=40,
                                 tpot_slo=2.0) == cmp,
    }
    return all(checks.values()), {"compare": cmp, "checks": checks,
                                  "config": {"n": 24, "rate": 0.8, "maxBatch": 4, "seed": seed}}


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["mock"], default="mock")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    ok, detail = _offline_invariants(seed=args.seed)
    report = {
        "benchmark": "serving-batching-sim",
        "mode": "mock-offline",
        "visibility": "public-aggregate",
        "claimStatus": "Simulation only — predicts the static→continuous batching uplift "
                       "shape; the measured number requires the real engine on GPU. Not a "
                       "served model.",
        "candidateOnly": True,
        "canClaimAGI": False,
        **detail,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    if ok:
        sp = detail["compare"]["throughputSpeedup"]
        print(f"SERVING SIM OFFLINE CORE VERIFIED ✓  (continuous throughput speedup ×{sp})")
        return 0
    print(f"SERVING SIM OFFLINE CORE FAILED ✗  failing: "
          f"{[k for k, v in detail['checks'].items() if not v]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Is a multi-seed ladder difference REAL, or just small-N sampling noise?

A multi-seed CI gate on a tiny eval (the 32-case internal ladder; religion N=6) can fail
for a statistical-power reason rather than a real-instability reason: at N=32 / ~65%
accuracy the binomial standard error per seed is ~8pp, so a seed-to-seed spread of a few
pp is INDISTINGUISHABLE from noise. This tool quantifies that, per domain and total:

  - observed seed-to-seed stdev vs the expected binomial SE at that N,
  - a verdict (within sampling noise -> underpowered; or exceeds it -> real variance),
  - the N required to resolve a target CI half-width (so you know how big the eval must be).

Pure stdlib, deterministic. It does not train or score; it audits whether the multi-seed
*claim* is even resolvable at the current eval size.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def binom_se(p: float, n: float) -> float:
    return math.sqrt(p * (1 - p) / n) if n > 0 else 0.0


def required_n(p: float, half_width: float, z: float = 1.96) -> int:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.ceil((z * z * p * (1 - p)) / (half_width * half_width))


def analyze_key(per_seed: list[tuple[int, int]]) -> dict[str, Any]:
    accs = [p / t for p, t in per_seed if t]
    n_avg = statistics.mean([t for _p, t in per_seed]) if per_seed else 0
    mean_acc = statistics.mean(accs) if accs else 0.0
    obs_stdev = statistics.stdev(accs) if len(accs) >= 2 else 0.0
    se = binom_se(mean_acc, n_avg)
    # "within noise" if the seed-to-seed spread does not exceed single-seed binomial SE.
    within_noise = obs_stdev <= se
    return {
        "meanAcc": round(mean_acc, 4),
        "nPerSeed": round(n_avg, 1),
        "observedSeedStdev": round(obs_stdev, 4),
        "binomialSE": round(se, 4),
        "ratio": round(obs_stdev / se, 2) if se else None,
        "withinSamplingNoise": within_noise,
        "requiredNForPlusMinus5pp": required_n(mean_acc, 0.05),
        "verdict": ("within sampling noise — multi-seed (in)stability is NOT resolvable at this N"
                    if within_noise else
                    "seed spread exceeds binomial SE — possible REAL instability (still confirm with more seeds)"),
    }


def analyze(seeds: list[dict[str, list[int]]]) -> dict[str, Any]:
    """seeds: list (one per seed) of {key: [passed, total]} incl. a 'total' key."""
    keys = sorted({k for s in seeds for k in s})
    out: dict[str, Any] = {"schema": "sophia.seed_stability.v1", "nSeeds": len(seeds), "byKey": {}}
    for k in keys:
        per_seed = [tuple(s[k]) for s in seeds if k in s]
        out["byKey"][k] = analyze_key(per_seed)
    underpowered = [k for k, v in out["byKey"].items() if v["withinSamplingNoise"]]
    out["underpoweredKeys"] = underpowered
    out["note"] = ("Keys flagged withinSamplingNoise cannot support a multi-seed CI claim at this "
                   "eval size — expand the eval (see requiredNForPlusMinus5pp) or use an external "
                   "benchmark with adequate N. A CI failure there is a power problem, not instability.")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds-json", required=True,
                    help='JSON list, one obj per seed: [{"total":[22,32],"religion":[1,6],...}, ...]')
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    seeds = json.loads(args.seeds_json)
    report = analyze(seeds)
    for k, v in report["byKey"].items():
        print(f"{k:12s} mean={v['meanAcc']:.3f} n={v['nPerSeed']:.0f} "
              f"obsStdev={v['observedSeedStdev']:.3f} binomSE={v['binomialSE']:.3f} -> {v['verdict']}")
    print(f"\nunderpowered keys: {report['underpoweredKeys']}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

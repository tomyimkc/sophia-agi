#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Measure the long-horizon autonomy curve: success rate vs task length.

The headline is the EFFECTIVE HORIZON — the longest task length still solved at
>=50%, judged by an external oracle (independent recomputation), not self-report.

    python tools/run_horizon_curve.py                       # built-in solvers (offline demo)
    python tools/run_horizon_curve.py --model ollama:llama3.2:3b --trials 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import horizon  # noqa: E402


def _print(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    print("length  successRate")
    for c in result["curve"]:
        print(f"  {c['length']:>3}      {c['successRate']:>6.1%}")
    print(f"effective horizon (>=50%): {result['effectiveHorizon']} steps")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="model spec to evaluate (else built-in demo solvers)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--lengths", default="1,2,4,8,16,32")
    args = ap.parse_args(argv)
    lengths = tuple(int(x) for x in args.lengths.split(",") if x.strip())

    if args.model:
        _print(args.model, horizon.horizon_curve(
            horizon.model_solver(args.model), lengths=lengths, trials=args.trials))
    else:
        _print("perfect solver (sanity: horizon should be max length)",
               horizon.horizon_curve(horizon.perfect_solver, lengths=lengths, trials=args.trials))
        _print("noisy solver (10% per-step error — realistic decay)",
               horizon.horizon_curve(horizon.noisy_solver(0.10), lengths=lengths, trials=args.trials))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

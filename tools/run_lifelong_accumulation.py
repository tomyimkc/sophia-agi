#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline runner for the closed-loop lifelong-accumulation benchmark.

Prints a human-readable summary of the honest net-capability measurement and,
with ``--out PATH``, writes the full JSON report. Deterministic; no network/LLM.

    python tools/run_lifelong_accumulation.py
    python tools/run_lifelong_accumulation.py --episodes 16 --out report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.lifelong_accumulation import (  # noqa: E402
    accumulates_cleanly,
    make_lifelong_stream,
    run_accumulation,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--out", type=str, default=None, help="write full JSON report here")
    args = ap.parse_args(argv)

    episodes = make_lifelong_stream(seed=args.seed, n_episodes=args.episodes)
    report = run_accumulation(episodes, seed=args.seed)

    curve = report["netCapabilityCurve"]
    print(f"schema={report['schema']} candidateOnly={report['candidateOnly']} "
          f"validated={report['validated']}")
    print("episode  graphCorrect  baselineCorrect  graphAcc  baselineAcc")
    for row in curve:
        print(f"{row['episode']:>7}  {row['graphCorrectCumulative']:>12}  "
              f"{row['baselineCorrectCumulative']:>15}  "
              f"{row['graphAccuracyCumulative']:>8}  {row['baselineAccuracyCumulative']:>11}")
    print()
    print(f"finalGraphCorrect    = {report['finalGraphCorrect']}")
    print(f"finalBaselineCorrect = {report['finalBaselineCorrect']}")
    print(f"accumulates          = {report['accumulates']}")
    print(f"unintendedForgetting = {report['unintendedForgetting']}")
    print(f"deliberateUnlearning = {report['deliberateUnlearning']}")
    print(f"cage                 = {report['cage']['committed']} committed / "
          f"{report['cage']['rejected']} rejected "
          f"({report['cage']['poisonRejected']} poison), "
          f"breaches={report['cage']['breaches']}")
    print(f"controlFlowGap       = {report['controlFlowGap']}")
    print(f"citations            = {len(report['citations'])} version-tagged facts")
    weakest = report["learningPriorities"][0] if report["learningPriorities"] else None
    if weakest:
        print(f"weakestDomain        = {weakest['domain']} "
              f"(competence={weakest['competence']}, deficit={weakest['deficit']})")
    print(f"accumulatesCleanly   = {accumulates_cleanly(report)}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                                  encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

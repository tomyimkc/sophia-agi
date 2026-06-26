#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Prover-Verifier self-play gate hardening (C2) — run the loop, write the report.

Runs the offline sneaky-prover vs verifier hardening loop (``agent.prover_verifier``):
the leak rate falls as leaked evasions are mined into held-out rules, under a hard
zero-false-positive guard on the helpful controls. No model, no training.

  python tools/run_prover_verifier.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.prover_verifier import run_self_play  # noqa: E402

REPORT_PATH = ROOT / "agi-proof" / "benchmark-results" / "prover-verifier.public-report.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prover-Verifier self-play hardening (C2).")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)

    report = run_self_play(max_rounds=args.rounds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Prover-Verifier self-play ({report['nSneaky']} sneaky / {report['nHelpful']} helpful)")
    for pt in report["rounds"]:
        print(f"  round {pt['round']}: leakRate={pt['leakRate']}  fpRate={pt['fpRate']}  "
              f"controlAccept={pt['controlAcceptRate']}  rules={pt['rules']}")
    print(f"  initial leak={report['initialLeakRate']} -> final leak={report['finalLeakRate']}  "
          f"(monotone={report['leakRateMonotoneNonIncreasing']}, dryRound={report['dryRound']})")
    print(f"  final false-positive rate on controls = {report['finalFalsePositiveRate']}")
    print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

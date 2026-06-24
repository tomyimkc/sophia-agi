#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the SSIL orchestrator over demo candidates and emit one signed-style artifact.

This is the runnable bounded-self-improvement loop: each candidate self-modification
must clear every wired gate (G2 reward-isolation, G4 plasticity, G5 honeypots, G6
corrigibility). The loop fails closed — any gate's reject blocks promotion. The
artifact shows, in one file, a genuine improvement promoted and reward-hack /
Goodhart / incorrigible candidates rejected.

Output is candidate infrastructure only: ``candidateOnly: true``,
``level3Evidence: false``, ``canClaimAGI: false``. This is NOT a proof of AGI/RSI;
it is a bounded, verifier-gated loop whose scorer is outside the optimizer's reach.

See docs/11-Platform/Safe-Self-Improvement-Loop.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil import demo_ssil_report  # noqa: E402

DEFAULT_OUT = ROOT / "agi-proof" / "self-extension" / "ssil-loop.public-report.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="SSIL bounded self-improvement loop")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output report path")
    ap.add_argument("--print", action="store_true", help="print the report to stdout")
    args = ap.parse_args()

    report = demo_ssil_report()
    ok = all(report["invariants"].values())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"SSIL loop: promoted={report['promoted']} rejected={report['rejected']}")
    print(f"SSIL invariants: {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

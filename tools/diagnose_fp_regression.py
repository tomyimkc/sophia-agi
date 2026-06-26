#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Diagnose an RLVR adapter's protected-integrity regression: name the TRUE cases that
flipped to false-positive (base did not, adapter did).

The seed-1 shaped-reward replication rejected on a -0.03 integrity regression, but the
committed aggregate only records the aggregate FP delta. This tool reads a FULL eval report
(the one the pod produces, with `falsePositiveRegressions` or per-case `rows`) and prints
exactly which true cases the adapter started getting wrong — so the next reward-shaping
change targets the real failure mode instead of guessing.

Usage:
  python3 tools/diagnose_fp_regression.py <report.json> [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def regressions_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the false-positive regressions: prefer the precomputed field, else derive it
    from the per-case `rows` the eval emits. Empty if neither is present."""
    pre = report.get("falsePositiveRegressions")
    if isinstance(pre, list):
        return pre
    rows = report.get("rows")
    if isinstance(rows, dict) and isinstance(rows.get("base"), list) and isinstance(rows.get("adapter"), list):
        from tools.eval_rlvr_adapter import false_positive_regressions

        return false_positive_regressions(rows["base"], rows["adapter"])
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", help="path to a full eval_rlvr_adapter report JSON")
    ap.add_argument("--json", action="store_true", help="print the regressions as JSON")
    args = ap.parse_args(argv)

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    regs = regressions_from_report(report)

    fp_delta = (report.get("delta") or {}).get("trueFalsePositiveRate")
    if args.json:
        print(json.dumps({"fpDelta": fp_delta, "count": len(regs), "regressions": regs}, ensure_ascii=False, indent=2))
        return 1 if regs else 0

    print(f"trueFalsePositiveRate delta: {fp_delta}")
    if not regs:
        print("no per-case false-positive regressions found "
              "(report lacks rows/falsePositiveRegressions, or none flipped).")
        return 0
    print(f"{len(regs)} TRUE case(s) the adapter regressed to a false-positive:")
    for r in regs:
        denied = " [denied-on-true-case]" if r.get("adapterDeniedOnTrueCase") else ""
        print(f"  - {r['case_id']} ({r.get('work')}): base {r['baseReward']} -> adapter {r['adapterReward']}{denied}")
        if r.get("adapterCompletion"):
            print(f"      adapter said: {str(r['adapterCompletion'])[:160]}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

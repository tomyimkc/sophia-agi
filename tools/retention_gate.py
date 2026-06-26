#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Retention GO/NO-GO gate — fail-closed guardrail on catastrophic forgetting.

Reads an M3 retention report (produced by tools/pilot_gemma3_run.py --retention: base-vs-adapter
on the held-out generality probe, deterministically scored) and decides whether an adapter is
promotable on the STABILITY axis (pre-registered criterion #3). The contract: an adapter that
forgets general capability must NOT be silently promoted.

    python3 tools/retention_gate.py agi-proof/benchmark-results/wisdom-market/M3-pilot-retention-eval.json
    # exit 0 = GO (retains), exit 3 = NO-GO (forgetting), exit 2 = unreadable/invalid report

Decision (default --tolerance 0.05):
  GO    if delta >= -tolerance            (point-estimate criterion #3 holds)
  NO-GO otherwise; the report also carries delta_ci95 + forgetting_established (whole CI < -tol),
        which distinguishes "established forgetting" from "point-estimate miss within noise".
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def gate(report: dict, tolerance: float) -> dict:
    delta = report.get("delta")
    ci = report.get("delta_ci95") or [None, None]
    base = report.get("base_accuracy")
    adapter = report.get("adapter_accuracy")
    n = report.get("nTasks")
    if delta is None:
        return {"ok": False, "verdict": "NO-GO", "reason": "report has no delta", "code": 2}
    go = delta >= -tolerance
    established = bool(report.get("forgetting_established"))
    return {
        "ok": go, "verdict": "GO" if go else "NO-GO",
        "delta": delta, "delta_ci95": ci, "base_accuracy": base, "adapter_accuracy": adapter,
        "nTasks": n, "tolerance": -tolerance, "forgetting_established": established,
        "reason": (f"retains: delta {delta:+.4f} >= -{tolerance}" if go else
                   f"FORGETTING: delta {delta:+.4f} < -{tolerance}"
                   + (" (CI fully below threshold — established)" if established
                      else " (point estimate; CI may straddle threshold — widen N to confirm)")),
        "code": 0 if go else 3,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="M3 retention eval JSON")
    ap.add_argument("--tolerance", type=float, default=0.05, help="max allowed drop vs base (criterion #3)")
    args = ap.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "verdict": "NO-GO", "reason": f"unreadable report: {exc}", "code": 2}))
        return 2
    result = gate(report, args.tolerance)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"RETENTION GATE: {result['verdict']} — {result['reason']}", file=sys.stderr)
    return int(result["code"])


if __name__ == "__main__":
    raise SystemExit(main())

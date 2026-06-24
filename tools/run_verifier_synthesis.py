#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the verifier-synthesis demonstration and check its falsifiable invariants.

Synthesises verifiers for tasks it has never seen, validates them against an
independent oracle (meta-verification), and contrasts that with the same run with
meta-verification turned OFF. Also reports calibrated abstention for the
unverifiable case. Exits non-zero if any invariant fails, so CI gates the claim.

    python tools/run_verifier_synthesis.py [--seed N] [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.synthesis_eval import run_demo  # noqa: E402


def _fmt(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = ap.parse_args()

    res = run_demo(seed=args.seed)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res["ok"] else 1

    w, wo = res["withMetaVerify"], res["withoutMetaVerify"]
    print("Verifier synthesis — synthesise a check, then verify the verifier")
    print("=" * 68)
    print("\nWITH meta-verification (validate before trusting):")
    print(f"  in-library  : {_fmt(w['inLibrary'])}")
    print(f"  out-of-lib  : {_fmt(w['outOfLibrary'])}")
    print("\nWITHOUT meta-verification (ablation — trust every fitted check):")
    print(f"  in-library  : {_fmt(wo['inLibrary'])}")
    print(f"  out-of-lib  : {_fmt(wo['outOfLibrary'])}")
    print("\nCalibrated abstention (competence where no verifier exists):")
    print(f"  {_fmt(res['calibration'])}")
    print("\nFalsifiable invariants:")
    for name, ok in res["invariants"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print("\n" + ("ALL INVARIANTS HOLD" if res["ok"] else "INVARIANT FAILURE"))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

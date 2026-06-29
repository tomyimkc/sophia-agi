#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Conformal auto-approval calibration — the price-of-guarantee curve (AATS experiment 3).

Thesis claim (docs/research/ai-auto-approval-thesis.md §4-A, idea 2): an auto-approver
should ABSTAIN (escalate to a human) outside the region where it can bound its
false-approval rate, and conformal prediction gives a distribution-free way to do
that — at the cost of an explicit escalation region. This harness does NOT reinvent
conformal prediction: it drives the repo's existing split-conformal machinery
(``agent.conformal_gate`` + ``tools.fit_conformal_policy``) and reports the operating
curve in auto-approval terms.

For a sweep of risk levels alpha it fits on a calibration split and measures, on a
held-out split:

  * ``approvalRate``   — fraction auto-approved (conformal "answer");
  * ``escalationRate`` — fraction sent to a human (1 - approvalRate) = the PRICE;
  * ``falseApprovalRate`` — approved-but-incorrect / approved (the thing we bound);
  * ``validityHolds``  — whether the held-out coverage guarantee actually held.

It then picks the operating point with the SMALLEST escalation rate whose held-out
false-approval rate is <= a target epsilon — "to guarantee false-approval <= eps you
must escalate at least this fraction". That sentence is the whole experiment.

    python tools/aats_conformal_calibration.py --synthetic 600 --target-false-approve 0.02
    python tools/aats_conformal_calibration.py --data data/outcomes.labeled.jsonl --target-false-approve 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.conformal_gate import evaluate_policy, fit_conformal_policy, load_jsonl  # noqa: E402
from tools.fit_conformal_policy import _correct_coverage, _split, synthetic_rows  # noqa: E402

REPORT_PATH = ROOT / "agi-proof" / "aats" / "conformal-calibration.public-report.json"
ALPHA_SWEEP = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3)


def calibration_curve(rows: list[dict], *, holdout: float = 0.5,
                      alphas=ALPHA_SWEEP) -> list[dict]:
    """Fit at each alpha on the calibration split; measure the auto-approval operating
    point on the held-out split. 'normal' risk bucket (the whole set if unbucketed)."""
    calib, test = _split(rows, holdout=holdout)
    curve = []
    for a in alphas:
        policy = fit_conformal_policy(calib, alpha=a)
        ev = evaluate_policy(policy, test)
        validity = _correct_coverage(policy, test)
        m = ev["metrics"]
        curve.append({
            "alpha": a,
            "threshold": policy.threshold,
            "approvalRate": m["coverage"],
            "escalationRate": round(1.0 - m["coverage"], 4),
            "falseApprovalRate": m["falseAnswerRate"],
            "selectiveAccuracy": m["selectiveAccuracy"],
            "validityHolds": (validity or {}).get("holds"),
        })
    return curve


def choose_operating_point(curve: list[dict], *, target_false_approve: float) -> "dict | None":
    """Smallest-escalation operating point whose held-out false-approval rate <= target."""
    feasible = [p for p in curve if p["falseApprovalRate"] <= target_false_approve]
    if not feasible:
        return None
    return min(feasible, key=lambda p: p["escalationRate"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Conformal auto-approval calibration curve (AATS exp 3).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="labeled rows JSONL {nonconformity, correct}")
    src.add_argument("--synthetic", type=int, metavar="N", help="N deterministic synthetic rows")
    ap.add_argument("--target-false-approve", type=float, default=0.02,
                    help="epsilon: max acceptable held-out false-approval rate")
    ap.add_argument("--holdout", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)

    if args.synthetic is not None:
        rows = synthetic_rows(args.synthetic)
        synthetic = True
    else:
        rows = [r for r in load_jsonl(args.data) if "correct" in r and "nonconformity" in r]
        synthetic = False
    if not rows:
        print(json.dumps({"error": "no labeled rows {nonconformity, correct}"}, indent=2))
        return 2

    curve = calibration_curve(rows, holdout=args.holdout)
    chosen = choose_operating_point(curve, target_false_approve=args.target_false_approve)
    report = {
        "schema": "sophia.aats_conformal_calibration.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "syntheticData": synthetic,
        "validated": False,
        "n": len(rows),
        "targetFalseApproveRate": args.target_false_approve,
        "curve": curve,
        "chosenOperatingPoint": chosen,
        "honestBound": ("Synthetic noisy-predictor rows — exercises the conformal price-of-guarantee "
                        "MACHINERY, not a Sophia capability. Replace with emit_outcome_records.py "
                        "--model output + >=3 runs for a result." if synthetic else
                        "Real labeled rows; candidate until >=2 judge families + >=3 runs + CI clear "
                        "the no-overclaim gate.") + " canClaimAGI false.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Conformal auto-approval calibration (synthetic={synthetic}, n={len(rows)})")
    print(f"  {'alpha':>6} {'approve':>8} {'escalate':>9} {'falseAppr':>10} {'valid':>6}")
    for p in curve:
        print(f"  {p['alpha']:>6.2f} {p['approvalRate']:>8.3f} {p['escalationRate']:>9.3f} "
              f"{p['falseApprovalRate']:>10.3f} {str(p['validityHolds']):>6}")
    if chosen:
        print(f"  -> to hold false-approval <= {args.target_false_approve}: escalate "
              f"{chosen['escalationRate']:.3f} (alpha={chosen['alpha']}, approve {chosen['approvalRate']:.3f})")
    else:
        print(f"  -> NO operating point meets false-approval <= {args.target_false_approve}: "
              f"escalate everything (no auto-approval is safe on this data)")
    print(f"Wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

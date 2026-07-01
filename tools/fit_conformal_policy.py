#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fit + validate a split-conformal abstention policy, persist it, and report (C1).

Turns labeled outcome records (``tools/emit_outcome_records.py``) into a fitted
:class:`agent.conformal_gate.ConformalPolicy` with a **held-out validity check** — the
honest core of the candidate. Split-conformal calibrates a nonconformity threshold
``tau`` on *correct* rows so a new correct point is accepted with probability >= 1-alpha;
the validity check measures whether that **coverage guarantee actually holds on a
held-out split** (fit on calib, measure correct-coverage on test).

Three jobs:

  1. ``--data PATH``    fit on real labeled rows {nonconformity, correct, risk?};
  2. ``--synthetic N``  fit on a deterministic synthetic set where nonconformity is a
                        genuine-but-noisy predictor of correctness — exercises and
                        validates the *machinery* offline. Marked ``syntheticData: true``;
                        NOT a Sophia capability result.
  3. persist + report   write the fitted policy to ``config/conformal_policy.json`` (the
                        artifact ``decide_conformal`` loads) and a candidate report under
                        ``agi-proof/benchmark-results/``.

    python tools/fit_conformal_policy.py --synthetic 400          # offline machinery check
    python tools/fit_conformal_policy.py --data data/outcomes.labeled.jsonl --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.conformal_gate import (  # noqa: E402
    ConformalPolicy,
    evaluate_policy,
    fit_conformal_policy,
    load_jsonl,
)

POLICY_PATH = ROOT / "config" / "conformal_policy.json"
REPORT_PATH = ROOT / "agi-proof" / "benchmark-results" / "conformal-policy.public-report.json"
DEFAULT_ALPHAS = (0.05, 0.1, 0.2)


def synthetic_rows(n: int, *, seed: int = 1729, sep: float = 8.0) -> list[dict]:
    """Deterministic rows where P(correct) decreases with nonconformity (noisy signal).

    A realistic calibration set: the score separates correct from incorrect, but
    imperfectly — so the conformal guarantee is non-trivial. Reproducible from ``seed``;
    NOT real data (``syntheticData: true`` in the report). ``sep`` is the logistic
    steepness (higher = cleaner separation between correct and incorrect); the default
    8.0 preserves the original behaviour and is used by the standard report.
    """
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        s = rng.random()  # nonconformity in [0,1]
        # logistic: low nonconformity -> likely correct; sep controls separation.
        p_correct = 1.0 / (1.0 + math.exp(sep * (s - 0.5)))
        correct = rng.random() < p_correct
        risk = "high" if s > 0.66 else "normal"
        rows.append({"id": f"syn{i}", "risk": risk, "nonconformity": round(s, 6), "correct": correct})
    return rows


def _split(rows: list[dict], *, holdout: float = 0.5) -> tuple[list[dict], list[dict]]:
    """Deterministic interleaved split (rows are exchangeable by construction)."""
    calib, test = [], []
    cut = max(1, int(round(1.0 / max(holdout, 1e-9))))
    for i, r in enumerate(rows):
        (test if (i % cut == 0) else calib).append(r)
    return (calib or rows), (test or rows)


def _correct_coverage(policy: ConformalPolicy, rows: list[dict]) -> "dict | None":
    """Fraction of CORRECT held-out rows that are ACCEPTED — the conformal guarantee.

    Split conformal guarantees a new correct point is accepted w.p. >= 1-alpha. This
    measures that coverage on held-out data and flags whether it meets the target within
    finite-sample slack ``2*sqrt(1/n_correct)`` (a generous Hoeffding-style band).
    """
    correct = [r for r in rows if bool(r.get("correct"))]
    if not correct:
        return None
    accepted = sum(1 for r in correct if policy.decide(float(r["nonconformity"]))["verdict"] == "answer")
    cov = accepted / len(correct)
    target = policy.target_coverage
    slack = 2.0 * math.sqrt(1.0 / len(correct))
    return {
        "nCorrect": len(correct),
        "heldOutCorrectCoverage": round(cov, 4),
        "targetCoverage": target,
        "slack": round(slack, 4),
        "holds": bool(cov >= target - slack),
    }


def fit_and_validate(rows: list[dict], *, alpha: float, holdout: float = 0.5) -> dict:
    calib, test = _split(rows, holdout=holdout)
    buckets = sorted({r.get("risk", "normal") for r in rows})
    by_risk = {}
    all_hold = True
    for risk in buckets:
        policy = fit_conformal_policy(calib, alpha=alpha, risk_bucket=risk)
        test_bucket = [r for r in test if r.get("risk", "normal") == risk] or test
        ev = evaluate_policy(policy, test_bucket)
        validity = _correct_coverage(policy, test_bucket)
        if validity is not None and not validity["holds"]:
            all_hold = False
        by_risk[risk] = {
            "policy": policy.to_dict(),
            "heldOut": ev["metrics"],
            "validity": validity,
        }
    return {"alpha": alpha, "byRisk": by_risk, "validityHolds": all_hold,
            "nCalib": len(calib), "nTest": len(test)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit + validate a split-conformal abstention policy (C1).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="labeled outcome records JSONL {nonconformity, correct}")
    src.add_argument("--synthetic", type=int, metavar="N", help="fit on N deterministic synthetic rows")
    ap.add_argument("--alpha", type=float, default=0.1, help="risk level for the persisted policy")
    ap.add_argument("--holdout", type=float, default=0.5)
    ap.add_argument("--persist", action="store_true", help="write the fitted policy to config/")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)

    if args.synthetic is not None:
        rows = synthetic_rows(args.synthetic)
        synthetic = True
    else:
        rows = load_jsonl(args.data)
        synthetic = False
    labeled = [r for r in rows if "correct" in r and "nonconformity" in r]
    if not labeled:
        print(json.dumps({"error": "no labeled rows with {nonconformity, correct}; "
                          "run emit_outcome_records.py --model first"}, indent=2))
        return 2

    sweep = [fit_and_validate(labeled, alpha=a, holdout=args.holdout) for a in DEFAULT_ALPHAS]
    # The persisted policy is fit on ALL labeled rows at the chosen alpha (no held-out).
    final_by_risk = {
        risk: fit_conformal_policy(labeled, alpha=args.alpha, risk_bucket=risk).to_dict()
        for risk in sorted({r.get("risk", "normal") for r in labeled})
    }
    report = {
        "schema": "sophia.conformal_policy_report.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "syntheticData": synthetic,
        "validated": False,
        "n": len(labeled),
        "persistAlpha": args.alpha,
        "alphaSweep": sweep,
        "persistedPolicyByRisk": final_by_risk,
        "honestBound": (
            "Synthetic data — demonstrates the conformal machinery (fit -> coverage "
            "guarantee -> held-out validity), NOT a Sophia capability. Replace with "
            "emit_outcome_records.py --model output + a third-party pack + >=3 runs for a "
            "result." if synthetic else
            "Real labeled rows; still candidate until >=2 judge families + >=3 runs + CI "
            "and a third-party pack clear the no-overclaim gate."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.persist:
        # Persist the 'normal' bucket (or the only bucket) as the live artifact.
        chosen = final_by_risk.get("normal") or next(iter(final_by_risk.values()))
        POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        POLICY_PATH.write_text(json.dumps(chosen, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Fitted conformal policy (synthetic={synthetic}, n={len(labeled)})")
    for s in sweep:
        flags = "VALID" if s["validityHolds"] else "FAILED-VALIDITY"
        print(f"  alpha={s['alpha']:.2f}  held-out validity: {flags}")
    print(f"Wrote {(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out)}")
    if args.persist:
        print(f"Persisted policy -> {POLICY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

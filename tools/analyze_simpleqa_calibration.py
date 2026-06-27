#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Headline-grade analysis of a SimpleQA calibration run (C1/C3).

Reads the detailed per-row JSONL from ``run_simpleqa_calibration.py`` (all confidence
signals + every grader's verdict) and computes the no-overclaim evidence:

  - inter-grader Cohen's kappa on the ternary A/B/C labels (>=2 grader families);
  - per-signal AUROC for predicting correctness (attempted rows);
  - risk-coverage curve with a BOOTSTRAP 95% CI on the selective-accuracy LIFT vs the
    answer-everything baseline — the headline claim, with its CI.

Numpy-free, deterministic bootstrap (fixed seed). Writes a public report.

  python tools/analyze_simpleqa_calibration.py
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DET = ROOT / "agi-proof" / "benchmark-results" / "real-model" / "simpleqa" / "simpleqa.detail.deepseek.jsonl"
OUT = ROOT / "agi-proof" / "benchmark-results" / "real-model" / "simpleqa" / "HEADLINE.public-report.json"
SIGNALS = {"stated": "stated", "selfcons": "selfcons", "logprob": "logprob_conf"}


def cohen_kappa(a: list, b: list) -> "float | None":
    """Cohen's kappa for two raters over categorical labels (numpy-free)."""
    pairs = [(x, y) for x, y in zip(a, b) if x in "ABC" and y in "ABC"]
    if not pairs:
        return None
    n = len(pairs)
    po = sum(x == y for x, y in pairs) / n
    cats = set(a) | set(b)
    pe = 0.0
    for c in cats:
        pa = sum(x == c for x, _ in pairs) / n
        pb = sum(y == c for _, y in pairs) / n
        pe += pa * pb
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


def auroc(pos: list, neg: list) -> "float | None":
    if not pos or not neg:
        return None
    wins = ties = 0
    for x in pos:
        for y in neg:
            wins += x > y
            ties += x == y
    return round((wins + 0.5 * ties) / (len(pos) * len(neg)), 4)


def risk_coverage(rows: list, key: str, covs=(1.0, 0.5, 0.3, 0.2, 0.1)) -> dict:
    rs = sorted([r for r in rows if r.get(key) is not None], key=lambda r: -r[key])
    n = len(rs)
    out = {}
    for c in covs:
        k = max(1, int(n * c))
        out[f"cov{int(c*100)}"] = round(sum(r["correct"] for r in rs[:k]) / k, 4)
    return out


def bootstrap_lift(rows: list, key: str, *, cov=0.2, B=3000, seed=13) -> dict:
    """Bootstrap 95% CI on (selectiveAcc@cov - overallAcc) over attempted rows."""
    rs = [r for r in rows if r.get(key) is not None]
    n = len(rs)
    rng = random.Random(seed)
    lifts, sel = [], []
    for _ in range(B):
        samp = [rs[rng.randrange(n)] for _ in range(n)]
        overall = sum(r["correct"] for r in samp) / n
        samp.sort(key=lambda r: -r[key])
        k = max(1, int(n * cov))
        s = sum(r["correct"] for r in samp[:k]) / k
        sel.append(s)
        lifts.append(s - overall)
    lifts.sort(); sel.sort()
    lo, hi = lifts[int(0.025 * B)], lifts[int(0.975 * B)]
    return {"coverage": cov, "selectiveAccMean": round(sum(sel) / B, 4),
            "liftMean": round(sum(lifts) / B, 4), "liftCI95": [round(lo, 4), round(hi, 4)],
            "liftExcludesZero": bool(lo > 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Headline analysis of a SimpleQA calibration run.")
    ap.add_argument("--detail", type=Path, default=DET)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.detail.read_text().splitlines() if l.strip()]
    graders = sorted({g for r in rows for g in (r.get("grades") or {})})
    overall_acc = round(sum(r["correct"] for r in rows) / len(rows), 4)
    attempted = [r for r in rows if r["action"] == "answer"]

    # inter-grader kappa (first two grader families)
    kappa = None
    if len(graders) >= 2:
        ga = [(r.get("grades") or {}).get(graders[0], "?") for r in rows]
        gb = [(r.get("grades") or {}).get(graders[1], "?") for r in rows]
        kappa = cohen_kappa(ga, gb)

    signals = {}
    for name, key in SIGNALS.items():
        cor = [r[key] for r in attempted if r.get(key) is not None and r["correct"]]
        wr = [r[key] for r in attempted if r.get(key) is not None and not r["correct"]]
        signals[name] = {"auroc": auroc(cor, wr),
                         "riskCoverage": risk_coverage(attempted, key),
                         "bootstrapLift@20": bootstrap_lift(attempted, key, cov=0.2)}

    best = max(signals, key=lambda s: signals[s]["auroc"] or 0)
    report = {
        "schema": "sophia.simpleqa_headline.v1", "candidateOnly": True, "level3Evidence": False,
        "syntheticData": False, "externalPublicBenchmark": "google/simpleqa-verified",
        "subjectModel": "deepseek-chat", "graders": graders,
        "n": len(rows), "nAttempted": len(attempted), "overallAccuracy": overall_acc,
        "accuracyAttempted": round(sum(r["correct"] for r in attempted) / len(attempted), 4) if attempted else 0,
        "interGraderCohenKappa": kappa,
        "kappaMeetsBar": (kappa is not None and kappa >= 0.40),
        "signals": signals, "bestSignal": best,
        "validated": bool(kappa is not None and kappa >= 0.40
                          and signals[best]["bootstrapLift@20"]["liftExcludesZero"]),
        "honestBound": (
            "Real EXTERNAL public benchmark (SimpleQA Verified), no self-authored data. "
            "Headline = selective-accuracy lift of the best confidence signal with a bootstrap "
            "95% CI, plus inter-grader Cohen's kappa on the ternary labels. Single subject model, "
            "single self-consistency sampling round (K samples) with bootstrap CI in lieu of 3 "
            "separate API re-runs (the temp=0 label is deterministic). canClaimAGI stays false."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"SimpleQA headline (n={len(rows)}, attempted={len(attempted)}, acc={overall_acc})")
    print(f"  inter-grader Cohen kappa = {kappa}  (>=0.40 bar: {report['kappaMeetsBar']})  graders={graders}")
    for name, s in signals.items():
        b = s["bootstrapLift@20"]
        print(f"  {name:9s} AUROC={s['auroc']}  selAcc@20%={b['selectiveAccMean']}  "
              f"lift={b['liftMean']} CI{b['liftCI95']} excl0={b['liftExcludesZero']}")
    print(f"  best signal = {best};  VALIDATED = {report['validated']}")
    print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

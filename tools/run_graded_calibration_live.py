#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Produce REAL labeled {confidence, correct} outcomes and fit the graded thresholds.

This is the stochastic-model run S3 was blocked on. For each attribution-benchmark case it:
  1. routes the question to an OKF page and answers it with a real model (the answer is
     stochastic — exactly what calibration needs);
  2. records the LIVE provenance confidence (`agent.grounded_confidence`) of the routed page;
  3. labels `correct` with the DETERMINISTIC trap scorer (`agent.benchmark_checks.score_case`)
     — no LLM judge, so the correctness label is objective;
then fits hi/lo with `tools/calibrate_thresholds.calibrate` and reports the data-driven
suggestion against the current honest default. Writes a candidate report; never mutates
`agent/graded_decision.DEFAULT_THRESHOLDS`.

Network + key required (not run in CI):
  OPENROUTER_API_KEY=... python tools/run_graded_calibration_live.py --model deepseek/deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "agi-proof" / "benchmark-results" / "graded-calibration-live.public-report.json"


def run(*, model: str, max_tokens: int = 400, limit: int | None = None) -> dict:
    from agent.benchmark_checks import load_json, score_case
    from agent.config import ROOT as CFG_ROOT, WIKI_DIR
    from agent.graded_decision import DEFAULT_THRESHOLDS, decide
    from agent.grounded_agent import grounded_answer
    from agent.openrouter_client import make_complete
    from okf.page import load_pages
    from tools.calibrate_thresholds import _balanced_accuracy, calibrate
    from tools.eval_rag_benchmark import all_cases

    pages = load_pages(WIKI_DIR)
    traditions = load_json(CFG_ROOT / "data" / "traditions.json")
    complete = make_complete(model=model, max_tokens=max_tokens)

    cases = all_cases()
    if limit:
        cases = cases[:limit]

    records: list[dict] = []
    for domain, case in cases:
        out = grounded_answer(case["question"], complete, pages=pages,
                              graded=True, confidence_from_sources=True)
        raw = out.get("rawAnswer", out["answer"])  # the model's real answer, pre-hedge label
        ok, reasons = score_case(case, raw, traditions)
        conf = out.get("graded", {}).get("confidence")
        records.append({
            "id": case["id"], "domain": domain, "target": out.get("target"),
            "policy": out["policy"], "action": out.get("graded", {}).get("action"),
            "confidence": conf, "correct": bool(ok),
        })
        print(f"  {case['id']:28s} target={out.get('target')} conf={conf} correct={ok}")

    labeled = [{"confidence": r["confidence"], "correct": r["correct"]}
               for r in records if r["confidence"] is not None]
    fit = calibrate(labeled) if labeled else {"error": "no confidence-bearing records"}
    default_bal = (round(_balanced_accuracy(labeled, DEFAULT_THRESHOLDS["hi"]), 4)
                   if labeled else None)

    n_correct = sum(1 for r in records if r["correct"])
    return {
        "benchmark": "graded-threshold live calibration (real model, deterministic scorer)",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "model": model,
        "n": len(records),
        "nLabeled": len(labeled),
        "accuracy": round(n_correct / len(records), 4) if records else None,
        "currentDefault": {**DEFAULT_THRESHOLDS, "balancedAccuracy": default_bal},
        "fit": fit,
        "records": records,
        "honestBound": ("Real stochastic-model answers labeled by the deterministic trap "
                        "scorer (no LLM judge). Fitted hi/lo is a CANDIDATE over a small, "
                        "self-authored in-domain set (n<=35) — it is NOT auto-applied to "
                        "DEFAULT_THRESHOLDS and is not a headline capability claim."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    report = run(model=args.model, max_tokens=args.max_tokens, limit=args.limit)
    fit = report["fit"]
    print(f"\nn={report['n']} labeled={report['nLabeled']} accuracy={report['accuracy']}")
    if "bestHi" in fit:
        print(f"  current default hi={report['currentDefault']['hi']} "
              f"balAcc={report['currentDefault']['balancedAccuracy']}")
        print(f"  FITTED (candidate) hi={fit['bestHi']} lo={fit['lo']} balAcc={fit['balancedAccuracy']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

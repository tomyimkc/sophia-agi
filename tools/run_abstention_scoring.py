#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Abstention-aware scoring (Kalai reform, C3) — reward IDK, penalise confident-wrong.

Re-scores a run's outcome records ({correct, action/verdict}) under the asymmetric
rubric (+1 correct / 0 abstain / -lambda wrong) and sweeps lambda to find the
break-even penalty above which fail-closed abstention beats always-guessing
(:mod:`agent.abstention_scoring`). Writes a candidate report; always reports the legacy
binary score alongside, never replacing it.

  python tools/run_abstention_scoring.py --data data/outcomes.labeled.jsonl
  python tools/run_abstention_scoring.py --synthetic 400   # offline machinery demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.abstention_scoring import lambda_sweep, score  # noqa: E402
from agent.conformal_gate import load_jsonl  # noqa: E402

REPORT_PATH = ROOT / "agi-proof" / "benchmark-results" / "abstention-scoring.public-report.json"


def _synthetic_decisions(n: int) -> list[dict]:
    """Synthetic {correct, action} via a fitted conformal policy over synthetic rows."""
    from agent.conformal_gate import fit_conformal_policy
    from agent.graded_decision import decide_conformal
    from tools.fit_conformal_policy import synthetic_rows

    rows = synthetic_rows(n)
    policy = fit_conformal_policy(rows, alpha=0.1)
    out = []
    for r in rows:
        conf = 1.0 - float(r["nonconformity"])
        d = decide_conformal(gate_passed=True, confidence=conf, policy=policy)
        out.append({"id": r["id"], "correct": bool(r["correct"]), "action": d["action"]})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Abstention-aware scoring + lambda sweep (C3).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="outcome records JSONL with {correct, action}")
    src.add_argument("--synthetic", type=int, metavar="N")
    ap.add_argument("--lambda", dest="lam", type=float, default=1.0, help="penalty for the headline score")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    args = ap.parse_args(argv)

    if args.synthetic is not None:
        records = _synthetic_decisions(args.synthetic)
        synthetic = True
    else:
        records = load_jsonl(args.data)
        synthetic = False
    labeled = [r for r in records if "correct" in r]
    if not labeled:
        print(json.dumps({"error": "no rows with `correct`; abstention scoring needs labels"}, indent=2))
        return 2

    headline = score(labeled, lam=args.lam)
    sweep = lambda_sweep(labeled)
    report = {
        "schema": "sophia.abstention_scoring_report.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "syntheticData": synthetic,
        "validated": False,
        "headline": headline,
        "lambdaSweep": sweep,
        "honestBound": sweep["honestBound"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Abstention-aware scoring (synthetic={synthetic}, n={headline['counts']['n']})")
    print(f"  abstention rate={headline['abstentionRate']}  selective acc={headline['selectiveAccuracy']}")
    print(f"  break-even lambda*={sweep['breakEvenLambda']}")
    for pt in sweep["curve"]:
        win = "abstain-wins" if pt["abstentionWins"] else "guess-wins"
        print(f"  lambda={pt['lambda']:.1f}  aware={pt['awareTotal']:.1f}  "
              f"always-answer={pt['alwaysAnswerTotal']:.1f}  -> {win}")
    print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibrate the graded router's hi/lo thresholds against the LIVE confidence signal.

Two honest paths, because a *production-optimal* fit needs labeled outcomes from a
stochastic model run (see `tools/calibrate_thresholds.py` — we never bake a curve fit on a
deterministic proxy into the defaults):

  --operating-curve  (default, offline, deterministic)
      Over the real OKF wiki corpus, compute the live provenance confidence
      (`agent.grounded_confidence.grounded_source_confidence`) for every page and sweep the
      `hi`/`lo` cut points, reporting the answer/hedge/abstain MIX at each operating point —
      the coverage-vs-conservatism curve an operator uses to pick thresholds against a target
      answer-coverage. This is a sensitivity analysis, NOT a claim of an optimal threshold.

  --data PATH        (the production fit)
      Given a JSONL/JSON of REAL {confidence, correct} outcomes (emit them with
      `--emit-records` below, once a stochastic-model run has labeled them), fit hi/lo by
      `tools/calibrate_thresholds.calibrate` and report the suggestion vs the current default.

  --emit-records PATH  (the loop-closer)
      Run `grounded_answer(confidence_from_sources=True)` over a benchmark with an injected
      model, writing {id, confidence, policy, correct?} records — the bridge from the live
      signal to a calibratable dataset. With the offline stub it has no `correct` labels (it
      cannot judge truth offline); point `--model` at a real backend to produce real labels.

Nothing here mutates `agent/graded_decision.DEFAULT_THRESHOLDS`. Candidate output only.

  python tools/calibrate_graded_thresholds.py                  # operating curve, OKF wiki
  python tools/calibrate_graded_thresholds.py --data runs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.graded_decision import DEFAULT_THRESHOLDS, decide  # noqa: E402

OUT_PATH = ROOT / "agi-proof" / "benchmark-results" / "graded-threshold-curve.public-report.json"


def _corpus_confidences(hops: int = 1) -> "list[float]":
    from agent.config import WIKI_DIR
    from agent.grounded_confidence import grounded_source_confidence
    from okf.page import load_pages

    pages = load_pages(WIKI_DIR)
    out: list[float] = []
    for p in pages:
        c = grounded_source_confidence(p.id, pages, hops=hops)
        if c is not None:
            out.append(c)
    return out


def _action_mix(confidences: "list[float]", hi: float, lo: float) -> dict:
    counts = {"answer": 0, "hedge": 0, "abstain": 0}
    for c in confidences:
        counts[decide(gate_passed=True, confidence=c, thresholds={"hi": hi, "lo": lo})["action"]] += 1
    n = len(confidences) or 1
    # Report a rounded partition. Rounding each component independently can produce
    # 1.0001/0.9999 on corpus sizes that do not divide cleanly into 4 decimals; make
    # the final bucket the residual so downstream tests/reports keep the partition
    # invariant without changing the underlying counts.
    answer = round(counts["answer"] / n, 4)
    hedge = round(counts["hedge"] / n, 4)
    abstain = round(1.0 - answer - hedge, 4)
    return {
        "hi": round(hi, 3), "lo": round(lo, 3),
        "answer": answer,
        "hedge": hedge,
        "abstain": abstain,
    }


def operating_curve(*, hops: int = 1, grid: "list[float] | None" = None) -> dict:
    confidences = _corpus_confidences(hops=hops)
    his = grid or [round(0.4 + 0.05 * i, 3) for i in range(9)]  # 0.40 .. 0.80
    lo = DEFAULT_THRESHOLDS["lo"]
    curve = [_action_mix(confidences, hi, min(lo, hi)) for hi in his]
    default_point = _action_mix(confidences, DEFAULT_THRESHOLDS["hi"], DEFAULT_THRESHOLDS["lo"])
    return {
        "benchmark": "graded-threshold operating curve (live provenance signal, OKF wiki)",
        "candidateOnly": True,
        "validated": False,
        "level3Evidence": False,
        "n": len(confidences),
        "loFixedAt": lo,
        "currentDefault": {**DEFAULT_THRESHOLDS, "mix": default_point},
        "curve": curve,
        "honestBound": ("Sensitivity analysis of the answer/hedge/abstain MIX vs the hi cut "
                        "point over the real corpus — an operating curve for choosing a "
                        "threshold against target coverage, NOT a fitted optimum. A "
                        "production-optimal hi/lo needs labeled {confidence, correct} "
                        "outcomes from a stochastic model run (use --data); offline truth "
                        "labels cannot be fabricated. Defaults are unchanged."),
    }


def fit_from_data(path: Path) -> dict:
    from tools.calibrate_thresholds import _load, calibrate

    records = _load(path)
    result = calibrate(records)
    result["source"] = str(path)
    result["note"] = ("Fitted from real labeled outcomes; reported as a CANDIDATE. Defaults "
                      "are NOT auto-updated — review before changing DEFAULT_THRESHOLDS.")
    return result


def emit_records(path: Path, *, model: str = "stub") -> dict:
    """Bridge the live signal to a calibratable dataset over the CPQA wiki episodes.

    With the offline stub model there is no truth oracle, so `correct` is omitted — the
    records carry {id, confidence, policy} ready to be labeled by a real run. This proves
    the live-signal -> calibration loop is wired without fabricating correctness labels.
    """
    from agent.config import WIKI_DIR
    from agent.grounded_agent import grounded_answer
    from okf.page import load_pages

    pages = load_pages(WIKI_DIR)
    episodes_path = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"
    queries: list[dict] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            for q in json.loads(line).get("queries", []):
                if q.get("type") == "recall":
                    queries.append(q)

    stub = (lambda s, u: "Per the corpus record, " + u[:80])  # deterministic, offline
    records = []
    for q in queries:
        out = grounded_answer(q["q"], stub, pages=pages, attribution_check=lambda a, b: True,
                              graded=True, confidence_from_sources=True)
        g = out.get("graded", {})
        records.append({"id": q["id"], "target": out.get("target"), "policy": out.get("policy"),
                        "confidence": g.get("confidence"), "action": g.get("action")})
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                    encoding="utf-8")
    labeled = sum(1 for r in records if r.get("confidence") is not None)
    return {"emitted": len(records), "withConfidence": labeled, "model": model, "path": str(path),
            "note": "No `correct` labels offline (stub cannot judge truth); point --model at a "
                    "real backend + add a gold scorer to produce calibratable outcomes."}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--operating-curve", action="store_true", help="(default) sweep hi over the corpus")
    ap.add_argument("--data", type=Path, help="fit hi/lo from real {confidence, correct} outcomes")
    ap.add_argument("--emit-records", type=Path, help="bridge live runs to a calibratable dataset")
    ap.add_argument("--model", default="stub")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.data:
        report = fit_from_data(args.data)
    elif args.emit_records:
        report = emit_records(args.emit_records, model=args.model)
    else:
        report = operating_curve(hops=args.hops)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json or args.data or args.emit_records:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        d = report["currentDefault"]
        print(f"OKF wiki — graded-threshold operating curve (n={report['n']}, lo={report['loFixedAt']})")
        print(f"  current default hi={d['hi']}: answer={d['mix']['answer']:.2f} "
              f"hedge={d['mix']['hedge']:.2f} abstain={d['mix']['abstain']:.2f}")
        print("  hi     answer  hedge  abstain")
        for p in report["curve"]:
            print(f"  {p['hi']:.2f}   {p['answer']:.2f}    {p['hedge']:.2f}   {p['abstain']:.2f}")
        print(f"Wrote {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibrate the graded-decision hedge threshold from labeled confidence data.

``agent/graded_decision.DEFAULT_THRESHOLDS`` ships honest defaults (hi=0.7,
lo=0.4) — *not* fitted numbers. This tool fits the ``hi`` cut point (the
hedge-vs-abstain boundary the wired graded route uses) to a labeled dataset of
``{confidence, correct}`` records by maximizing **balanced accuracy** of the
predicate ``confidence >= hi`` against ``correct``. ``lo`` is reported as the
high-recall floor (the largest threshold that still keeps every correct answer
surfaceable).

Producing the dataset requires running the guarded loop with a *stochastic* model
(self-consistency needs sampling variation), so this stays a separate, explicit
step — we never bake a curve fit on a deterministic mock into the defaults.

Usage:
  python tools/calibrate_thresholds.py --data labeled.jsonl
  python tools/calibrate_thresholds.py --demo            # synthetic separable data
Each input line/record: {"confidence": 0.0..1.0, "correct": true|false}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> "list[dict]":
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":  # JSON array
        return list(json.loads(text))
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _balanced_accuracy(records: "list[dict]", hi: float) -> float:
    """Balanced accuracy of (confidence >= hi) predicting `correct`."""
    tp = fn = fp = tn = 0
    for r in records:
        surfaced = float(r["confidence"]) >= hi
        correct = bool(r["correct"])
        if correct and surfaced:
            tp += 1
        elif correct and not surfaced:
            fn += 1
        elif not correct and surfaced:
            fp += 1
        else:
            tn += 1
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return (tpr + tnr) / 2.0


def calibrate(records: "list[dict]", *, grid: int = 101) -> dict:
    """Return the best hi/lo with the achieved balanced accuracy and a small curve."""
    if not records:
        return {"error": "no records", "n": 0}
    candidates = [i / (grid - 1) for i in range(grid)]
    curve = [{"hi": round(h, 4), "balancedAccuracy": round(_balanced_accuracy(records, h), 4)}
             for h in candidates]
    best = max(curve, key=lambda p: (p["balancedAccuracy"], -p["hi"]))
    best_hi = best["hi"]
    # lo = high-recall floor: largest threshold still <= every correct answer's confidence.
    correct_confs = [float(r["confidence"]) for r in records if bool(r["correct"])]
    lo = min(correct_confs) if correct_confs else 0.0
    lo = round(min(lo, best_hi), 4)
    return {
        "n": len(records),
        "bestHi": best_hi,
        "lo": lo,
        "balancedAccuracy": best["balancedAccuracy"],
        "currentDefault": {"hi": 0.7, "lo": 0.4},
        "curve": curve,
    }


def _demo_records() -> "list[dict]":
    # Separable synthetic set: correct answers are high-confidence, errors low.
    correct = [{"confidence": c / 100.0, "correct": True} for c in range(80, 101, 2)]
    wrong = [{"confidence": c / 100.0, "correct": False} for c in range(0, 41, 2)]
    return correct + wrong


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate graded-decision thresholds")
    parser.add_argument("--data", type=Path, help="JSON/JSONL of {confidence, correct}")
    parser.add_argument("--demo", action="store_true", help="use synthetic separable data")
    parser.add_argument("--grid", type=int, default=101)
    args = parser.parse_args(argv)

    if args.demo:
        records = _demo_records()
    elif args.data:
        records = _load(args.data)
    else:
        parser.error("provide --data PATH or --demo")
        return 2

    result = calibrate(records, grid=args.grid)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    raise SystemExit(main())

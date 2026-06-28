#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibration check: does the source-discipline adapter OVER-qualify clear-cut cases?

A high qualification rate on *contested* cases is only half of calibration. The failure mode it
hides is reflexive hedging: emitting qualification language ("disputed", "traditionally attributed",
"uncertain"...) on *settled* questions where a direct answer is correct. This reads the committed
base-vs-adapter answer files and reports the hedge-marker rate on contested vs clear-cut families,
so the over-qualification tax is measured, not assumed. Deterministic, offline, no API.

    python3 tools/calibration_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WM = ROOT / "agi-proof" / "benchmark-results" / "wisdom-market"

# Same hedge markers the M1 qualification scorer uses (generic phrases, not a proprietary format).
HEDGE = ["legendary", "traditionally attributed", "compiled", "disputed", "uncertain",
         "scholars disagree", "contested", "may be", "is attributed to", "not certain",
         "no single author", "傳說", "相傳", "存疑", "有爭議", "彙編", "未必", "歸於"]
CONTESTED = {"authorship_boundary", "contested_religion", "provenance_qa", "tradition_boundary",
             "false_attribution"}
# Settled, gold_route=allow -> hedging is MISCALIBRATION (the answer is not contested).
CLEARCUT = {"protected_history", "protected_religion", "settled_clearcut"}


def _hedges(t: str) -> bool:
    t = (t or "").lower()
    return any(x in t for x in HEDGE)


def _rate(rows, key):
    return round(sum(_hedges(r.get(key, "")) for r in rows) / len(rows), 4) if rows else None


def check(answers_path: Path) -> dict:
    rows = json.loads(answers_path.read_text(encoding="utf-8"))
    ct = [r for r in rows if r.get("task_family") in CONTESTED]
    cc = [r for r in rows if r.get("task_family") in CLEARCUT]
    out = {
        "answers": answers_path.name,
        "contested": {"n": len(ct), "base_hedge": _rate(ct, "base_answer"),
                      "adapter_hedge": _rate(ct, "adapter_answer")},
        "clearcut": {"n": len(cc), "base_hedge": _rate(cc, "base_answer"),
                     "adapter_hedge": _rate(cc, "adapter_answer")},
    }
    cc_a, cc_b = out["clearcut"]["adapter_hedge"], out["clearcut"]["base_hedge"]
    out["overqualification_lift_clearcut"] = (round(cc_a - cc_b, 4) if cc_a is not None and cc_b is not None else None)
    out["verdict"] = ("OVER-QUALIFIES clear-cut cases (calibration tax)"
                      if (out["overqualification_lift_clearcut"] or 0) > 0.10 else "calibrated")
    return out


def main() -> int:
    seeds = ["M3-pilot-answers-seed1.json", "M3-pilot-answers-seed2.json"]
    results = [check(WM / s) for s in seeds if (WM / s).exists()]
    report = {
        "metric": "hedge-marker rate on contested (should be HIGH) vs clear-cut (should be LOW)",
        "note": ("adapter intentionally hedges contested cases; the calibration question is whether it "
                 "ALSO hedges settled (clear-cut) ones. lift = adapter_hedge - base_hedge on clear-cut."),
        "perSeed": results,
        "boundary": "clear-cut n is small (~38); direction is robust across seeds, magnitude noisy.",
    }
    (WM / "calibration-check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print("=== CALIBRATION CHECK (over-qualification on clear-cut cases) ===")
    for r in results:
        print(f"  {r['answers']}: contested adapter {r['contested']['adapter_hedge']} (base "
              f"{r['contested']['base_hedge']}) | clear-cut adapter {r['clearcut']['adapter_hedge']} "
              f"(base {r['clearcut']['base_hedge']}) -> lift {r['overqualification_lift_clearcut']} "
              f"[{r['verdict']}]")
    print("wrote -> agi-proof/benchmark-results/wisdom-market/calibration-check.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

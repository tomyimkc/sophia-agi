#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Justice-Consistency benchmark for the Dikaiosyne gate (Role A) + an honest receipt.

Runs the pre-registered battery
(``agi-proof/benchmark-results/dikaiosyne/dikaiosyne_justice_battery.json``) through
``agent.dikaiosyne.assess_justice`` and scores the impartiality routing:

    should_impartial vs should_partial vs should_false_equivalence
    missed partiality    : optimal partial but predicted not-partial (unfairness missed)
    overflagged partiality: optimal impartial/false_equivalence but predicted partial

It then emits a GO/NO-GO RECEIPT in the SAME spirit as ``tools/claim_gate.py``.
By construction this is **NO-GO / candidate**: ONE deterministic judge (the gate
itself), author-written battery, no baseline contrast, no effect size with a CI. We
ship the instrument and a candidate result; we do NOT claim Dikaiosyne improves real
decisions. The artifact is deterministic (no timestamps) so re-runs reproduce
identical bytes and the CI drift gate stays green.

Exit 0 always (report generator). With ``--strict`` exits 3 on a NO-GO verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dikaiosyne import assess_justice  # noqa: E402

BATTERY = ROOT / "agi-proof" / "benchmark-results" / "dikaiosyne" / "dikaiosyne_justice_battery.json"
OUT = ROOT / "agi-proof" / "benchmark-results" / "dikaiosyne" / "dikaiosyne-justice-calibration.json"


def _decide(case: dict) -> dict:
    ctx = {}
    if case.get("hardBlock"):
        ctx["hardBlock"] = True
    return assess_justice(
        case.get("text", ""),
        irrelevant_class=case.get("irrelevantClass"),
        relevant_class=case.get("relevantClass"),
        context=ctx,
    ).to_dict()


def _score(battery: dict) -> dict:
    rows = []
    missed_partiality = 0
    overflagged_partiality = 0
    correct = 0
    for c in battery.get("cases", []):
        d = _decide(c)
        got, opt = d["verdict"], c["optimal"]
        ok = got == opt
        correct += int(ok)
        if opt == "partial" and got != "partial":
            missed_partiality += 1
        if opt in {"impartial", "false_equivalence"} and got == "partial":
            overflagged_partiality += 1
        rows.append({
            "id": c["id"], "quadrant": c.get("quadrant"), "optimal": opt,
            "verdict": got, "jq": d["jq"], "blockRespected": d["blockRespected"],
            "ok": ok, "reason": d["reason"],
        })
    n = len(rows)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "missedPartiality": missed_partiality,
        "overflaggedPartiality": overflagged_partiality,
        "cases": rows,
    }


def _receipt(scored: dict) -> dict:
    critical_failures = [
        "single_judge: scoring uses one deterministic judge, not >=2 independent judge families",
        "battery_not_external: cases are author-written, not external/human-authored and decontaminated",
        "no_effect_size_with_ci: routing accuracy is not an effect on real-world decisions with a CI excluding zero",
        "no_preregistered_baseline_comparison: no raw-agent (no-auditor) baseline contrast established",
        "relevant_class_labels_author_only: which swaps are relevant/irrelevant is author-labelled, not 2-family confirmed",
    ]
    verdict = "NO-GO" if critical_failures else "GO"
    return {
        "verdict": verdict,
        "promotable": verdict == "GO",
        "criticalFailures": critical_failures,
        "boundary": (
            "Dikaiosyne is candidate infrastructure. This receipt is NO-GO by design: "
            "it certifies the gate's deterministic routing on a pre-registered battery, "
            "NOT a claim that the justice gate improves real decisions. Promotion past "
            "candidate requires >=2 independent judge families (for the relevant/irrelevant "
            "labels), an external decontaminated battery, a real no-auditor baseline, and "
            "an effect on the partiality / false-equivalence rates whose CI excludes zero."
        ),
    }


def run() -> dict:
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    scored = _score(battery)
    return {
        "schema": "sophia.dikaiosyne_justice_calibration.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "battery": battery.get("schema"),
        "preregistered": battery.get("preregistered", False),
        "thresholds": battery.get("thresholds", {}),
        "score": scored,
        "receipt": _receipt(scored),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT), help="output JSON path")
    ap.add_argument("--strict", action="store_true", help="exit 3 on a NO-GO receipt")
    ap.add_argument("--print", dest="show", action="store_true", help="print summary")
    args = ap.parse_args()

    report = run()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    s, r = report["score"], report["receipt"]
    print(f"Dikaiosyne justice-calibration: n={s['n']} accuracy={s['accuracy']} "
          f"missedPartiality={s['missedPartiality']} overflaggedPartiality={s['overflaggedPartiality']} "
          f"-> RECEIPT {r['verdict']} (candidate)")
    if args.show:
        for row in s["cases"]:
            flag = "ok" if row["ok"] else "XX"
            print(f"  {flag} {row['id']:38} optimal={row['optimal']:18} got={row['verdict']:18} jq={row['jq']}")
    print(f"wrote {out}")

    if args.strict and not report["receipt"]["promotable"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

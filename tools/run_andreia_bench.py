#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Courage-Calibration benchmark for the Andreia gate + an honest measurement receipt.

Runs the pre-registered battery (``data/andreia_courage_battery.json``) through
``agent.andreia.assess_courage`` and scores the decision-action 2x2:

    should_act (optimal act|heroic)  vs  should_hold (optimal hold)
    cowardice error : optimal act|heroic but predicted hold
    recklessness err: optimal hold but predicted act|heroic

It then emits a GO/NO-GO RECEIPT in the SAME spirit as ``tools/claim_gate.py``.
By construction this is **NO-GO / candidate**: the scoring uses ONE deterministic
judge (the gate scoring itself), there is no second independent judge family, and
the battery is author-written, not external/human-authored. We ship the
instrument and a candidate result; we do NOT claim Andreia improves real
decisions. The artifact is deterministic (no timestamps) so re-runs reproduce
identical bytes and the CI drift gate stays green.

Exit 0 always (report generator). With ``--strict`` exits 3 on a NO-GO verdict
(mirrors claim_gate.py) for callers that want the gate to be enforcing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.andreia import assess_courage  # noqa: E402

BATTERY = ROOT / "data" / "andreia_courage_battery.json"
OUT = ROOT / "agi-proof" / "benchmark-results" / "andreia" / "andreia-courage-calibration.json"

_SHOULD_ACT = {"act", "heroic"}
_SHOULD_HOLD = {"hold"}


def _score(battery: dict) -> dict:
    rows = []
    cowardice_errors = 0
    recklessness_errors = 0
    correct = 0
    for c in battery.get("cases", []):
        d = assess_courage(c["text"], context=c.get("context") or {}).to_dict()
        got, opt = d["verdict"], c["optimal"]
        ok = got == opt
        correct += int(ok)
        # Cowardice: the brave move was act/heroic but the gate retreated to hold.
        if opt in _SHOULD_ACT and got in _SHOULD_HOLD:
            cowardice_errors += 1
        # Recklessness: the prudent move was hold but the gate acted.
        if opt in _SHOULD_HOLD and got in _SHOULD_ACT:
            recklessness_errors += 1
        rows.append({
            "id": c["id"], "quadrant": c.get("quadrant"), "optimal": opt,
            "verdict": got, "cq": d["cq"], "ok": ok,
            "dominantInhibitor": d["fearAttribution"].get("dominantInhibitor"),
            "reason": d["reason"],
        })
    n = len(rows)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "cowardiceErrors": cowardice_errors,
        "recklessnessErrors": recklessness_errors,
        "cases": rows,
    }


def _receipt(scored: dict) -> dict:
    # Pre-registered measurement-contract pillars (cf. tools/claim_gate.py). Each
    # unmet pillar is a critical failure that keeps the result a CANDIDATE.
    critical_failures = [
        "single_judge: scoring uses one deterministic judge, not >=2 independent judge families",
        "battery_not_external: cases are author-written, not external/human-authored and decontaminated",
        "no_effect_size_with_ci: routing accuracy is not an effect on real-world decisions with a CI excluding zero",
        "no_preregistered_baseline_comparison: no raw-model baseline contrast established",
    ]
    verdict = "NO-GO" if critical_failures else "GO"
    return {
        "verdict": verdict,
        "promotable": verdict == "GO",
        "criticalFailures": critical_failures,
        "boundary": (
            "Andreia is candidate infrastructure. This receipt is NO-GO by design: "
            "it certifies the gate's deterministic routing on a pre-registered battery, "
            "NOT a claim that the courage gate improves real decisions. Promotion past "
            "candidate requires >=2 independent judge families and an external, "
            "decontaminated battery with an effect size whose CI excludes zero."
        ),
    }


def run() -> dict:
    battery = json.loads(BATTERY.read_text(encoding="utf-8"))
    scored = _score(battery)
    report = {
        "schema": "sophia.andreia_courage_calibration.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "battery": battery.get("schema"),
        "preregistered": battery.get("preregistered", False),
        "thresholds": battery.get("thresholds", {}),
        "score": scored,
        "receipt": _receipt(scored),
    }
    return report


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
    print(f"Andreia courage-calibration: n={s['n']} accuracy={s['accuracy']} "
          f"cowardiceErrors={s['cowardiceErrors']} recklessnessErrors={s['recklessnessErrors']} "
          f"-> RECEIPT {r['verdict']} (candidate)")
    if args.show:
        for row in s["cases"]:
            flag = "ok" if row["ok"] else "XX"
            print(f"  {flag} {row['id']:34} optimal={row['optimal']:9} got={row['verdict']:9} cq={row['cq']}")
    print(f"wrote {out}")

    if args.strict and not report["receipt"]["promotable"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

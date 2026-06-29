#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophrosyne robustness probe — stress-tests the WEAKEST link, honestly.

The temperance gate routes well when its mean-deviation inputs (demand/expenditure/
marginalValue/appetite/budget) are supplied explicitly — the calibration battery is
16/16. But in real use (e.g. the conscience `consultTemperance` path) the gate must
DERIVE those signals from raw text + the intemperance detector. This probe measures
how well that derivation holds up. It changes NOTHING in the gate; it only measures
and reports, so the honest limit lands in the failure ledger instead of being hidden.

Two probes (deterministic, offline):

A. Derivation gap — run the gate on each battery case WITH explicit context (the
   routing baseline) and WITHOUT any context (fully derived from text), and compare
   both to the labelled optimal. A large explicit-minus-derived gap means the gate's
   quality is bounded by signal-estimation quality (especially the demand set-point
   `delta`), not by its routing logic.

B. Intemperance-detector paraphrase brittleness — meaning-preserving paraphrases of
   excess/deficiency wording that evade the regex detectors. The detection-rate drop
   quantifies how brittle the deterministic features are to wording they did not
   enumerate (the same model-gated limit Andreia documents for cowardice).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.intemperance_signals import detect_intemperance  # noqa: E402
from agent.sophrosyne import assess_temperance  # noqa: E402

RESULTS_DIR = ROOT / "agi-proof" / "benchmark-results" / "sophrosyne"
BATTERY_PATH = RESULTS_DIR / "sophrosyne_measure_battery.json"
OUT_PATH = RESULTS_DIR / "sophrosyne-robustness.json"

# Meaning-preserving paraphrases of intemperate wording the regex does NOT
# enumerate. Each ORIGINAL fires an intemperance signal; each PARAPHRASE says the
# same thing in words the regex misses. (Authored as a brittleness probe, not a
# training set; the context only varies wording.)
PARAPHRASE_PAIRS = [
    # excess (over-hedging)
    ("I think perhaps it maybe could possibly be the case.",
     "On balance my tentative sense, held loosely, is that it might conceivably hold.",
     {"demand": 0.4, "expenditure": 0.8, "marginalValue": 0.25}),
    # deficiency (truncation)
    ("The rest is left as an exercise. TODO",
     "I'll let you work out the remainder yourself another time.",
     {"demand": 0.75, "expenditure": 0.3, "marginalValue": 0.7}),
    # excess (filler/padding)
    ("It is important to note that, needless to say, due to the fact that this matters.",
     "Worth flagging, obviously, given how much weight this carries in practice.",
     {"demand": 0.4, "expenditure": 0.8, "marginalValue": 0.25}),
]


def _load_battery() -> dict:
    return json.loads(BATTERY_PATH.read_text(encoding="utf-8"))


def probe_derivation_gap(battery: dict) -> dict:
    rows = []
    explicit_ok = derived_ok = 0
    derived_dist: Counter = Counter()
    for c in battery["cases"]:
        opt = c["optimal"]
        expl = assess_temperance(c["text"], context=c.get("context") or {}).to_dict()
        der = assess_temperance(c["text"]).to_dict()  # no context -> fully derived
        explicit_ok += expl["verdict"] == opt
        derived_ok += der["verdict"] == opt
        derived_dist[der["verdict"]] += 1
        rows.append({
            "id": c["id"], "optimal": opt,
            "explicitVerdict": expl["verdict"], "explicitOk": expl["verdict"] == opt,
            "derivedVerdict": der["verdict"], "derivedOk": der["verdict"] == opt,
            "derivedForces": der["forces"], "derivedMq": der["mq"],
        })
    n = len(rows)
    return {
        "n": n,
        "explicitAgreement": round(explicit_ok / n, 4) if n else 0.0,
        "derivedAgreement": round(derived_ok / n, 4) if n else 0.0,
        "gap": round((explicit_ok - derived_ok) / n, 4) if n else 0.0,
        "derivedVerdictDistribution": dict(derived_dist),
        "cases": rows,
    }


def probe_paraphrase_brittleness() -> dict:
    rows = []
    orig_fired = para_fired = 0
    for original, paraphrase, ctx in PARAPHRASE_PAIRS:
        o = detect_intemperance(original, context=ctx)
        p = detect_intemperance(paraphrase, context=ctx)
        o_fire = o.verdict != "measure_clear"
        p_fire = p.verdict != "measure_clear"
        orig_fired += o_fire
        para_fired += p_fire
        rows.append({
            "original": original, "originalFired": o_fire, "originalVerdict": o.verdict,
            "paraphrase": paraphrase, "paraphraseFired": p_fire, "paraphraseVerdict": p.verdict,
            "evaded": o_fire and not p_fire,
        })
    n = len(rows)
    return {
        "n": n,
        "originalDetectionRate": round(orig_fired / n, 4) if n else 0.0,
        "paraphraseDetectionRate": round(para_fired / n, 4) if n else 0.0,
        "evasionRate": round(sum(r["evaded"] for r in rows) / n, 4) if n else 0.0,
        "cases": rows,
    }


def build_report() -> dict:
    battery = _load_battery()
    deriv = probe_derivation_gap(battery)
    para = probe_paraphrase_brittleness()
    return {
        "schema": "sophia.sophrosyne_robustness.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "canClaimAGI": False,
        "derivationGap": deriv,
        "paraphraseBrittleness": para,
        "finding": (
            f"Routing on EXPLICIT mean-deviation inputs is {deriv['explicitAgreement']:.0%}, but on "
            f"raw text the DERIVED routing is only {deriv['derivedAgreement']:.0%} — the gate's quality "
            "is bounded by signal-estimation quality (above all the demand set-point `delta`), not by "
            f"its routing logic. The regex intemperance detectors miss {para['evasionRate']:.0%} of "
            "meaning-preserving paraphrases. So the conscience `consultTemperance` integration is "
            "conservative on raw text BY DESIGN, and no claim is made that the gate is temperate on raw text."
        ),
        "boundary": (
            "Deterministic candidate diagnostic. It measures the gate's input-derivation limits; it does "
            "not modify the gate and is not AGI proof. The paraphrase-robust fix is model-gated — wire a "
            "real semantic/NLI/LLM-judge backend through detect_intemperance(semantic_backend=...) and a "
            "model-backed demand estimator, then re-run; tracked in agi-proof/failure-ledger.md, never "
            "tuned to the battery."
        ),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="Sophrosyne robustness probe (derivation gap + paraphrase brittleness)")
    ap.add_argument("--out", default=str(OUT_PATH), help="output JSON path")
    ap.add_argument("--print", dest="show", action="store_true", help="print per-case detail")
    args = ap.parse_args(argv)

    report = build_report()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    d, p = report["derivationGap"], report["paraphraseBrittleness"]
    print(f"Sophrosyne robustness: explicit={d['explicitAgreement']} derived={d['derivedAgreement']} "
          f"gap={d['gap']} | paraphrase evasion={p['evasionRate']} "
          f"(orig {p['originalDetectionRate']} -> para {p['paraphraseDetectionRate']})")
    print(f"derived verdict distribution: {d['derivedVerdictDistribution']}")
    if args.show:
        for r in d["cases"]:
            print(f"  {r['id']:34} optimal={r['optimal']:13} explicit={r['explicitVerdict']:13} derived={r['derivedVerdict']:13}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

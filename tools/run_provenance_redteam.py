#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Provenance red-team — measure the source-independence graph's DETECTION RATE on forged fixtures.

Runs okf.evidence_spec.independence_report over eval/provenance_redteam/forged_corpus.jsonl and
scores, per fixture, whether the graph's verdict matches the fixture's ``detect`` label:

  * detect == "collapse"    — the graph must report effectiveIndependentCount < the fixture's
                              claimedIndependent (it saw through the seeded fake-independence).
  * detect == "no-collapse" — an honest control; the graph must NOT under-count genuine origins
                              (effectiveIndependentCount == claimedIndependent). Getting a control
                              wrong is a FALSE POSITIVE, reported separately.

The reported detection rate is HONEST but SCOPED: it is measured ON THIS small red-team corpus of
forged provenance fixtures ONLY. It is NOT a claim about detecting real-world forged provenance —
the graph can only collapse origins the data DECLARES; undeclared shared provenance is out of
scope (see evidence_spec.json honestLimits).

Exit: 0 always (this is a measurement report on fixtures, not a build gate); 2 = unreadable input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okf import evidence_spec as es  # noqa: E402

CORPUS = ROOT / "eval" / "provenance_redteam" / "forged_corpus.jsonl"


def score(corpus_path: Path) -> dict:
    rows = [json.loads(ln) for ln in corpus_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    attacks = [r for r in rows if r.get("detect") == "collapse"]
    controls = [r for r in rows if r.get("detect") == "no-collapse"]

    detected = []      # attack fixtures where collapse was correctly seen
    missed = []        # attack fixtures where the graph failed to see the collapse
    false_positive = []  # control fixtures wrongly flagged as collapsed
    per_row = []

    for r in rows:
        rep = es.independence_report(r.get("sources") or [])
        eff = rep["effectiveIndependentCount"]
        claimed = int(r.get("claimedIndependent", eff))
        collapsed_flag = eff < claimed
        row = {"id": r.get("id"), "attackType": r.get("attackType"),
               "claimedIndependent": claimed, "effectiveIndependentCount": eff,
               "detectExpected": r.get("detect"), "collapsedDetected": collapsed_flag}
        per_row.append(row)
        if r.get("detect") == "collapse":
            (detected if collapsed_flag else missed).append(r.get("id"))
        else:  # no-collapse control
            if collapsed_flag:
                false_positive.append(r.get("id"))

    detection_rate = len(detected) / len(attacks) if attacks else 0.0
    fp_rate = len(false_positive) / len(controls) if controls else 0.0

    return {
        "experiment": "provenance-redteam-independence-detection",
        "corpus": str(corpus_path.relative_to(ROOT)),
        "nFixtures": len(rows),
        "nAttacks": len(attacks),
        "nControls": len(controls),
        "detectionRate": round(detection_rate, 4),
        "falsePositiveRate": round(fp_rate, 4),
        "detected": detected,
        "missed": missed,
        "falsePositives": false_positive,
        "perRow": per_row,
        "honestScope": ("detection rate measured on eval/provenance_redteam/forged_corpus.jsonl "
                        "fixtures ONLY; the graph only collapses DECLARED shared origins — "
                        "undeclared shared provenance is out of scope; NOT a real-world claim"),
        "canClaimAGI": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    args = ap.parse_args()
    if not args.corpus.exists():
        print(json.dumps({"experiment": "provenance-redteam", "reason": "corpus missing",
                          "code": 2}))
        return 2
    res = score(args.corpus)
    print(f"PROVENANCE RED-TEAM: detectionRate={res['detectionRate']} "
          f"(attacks={res['nAttacks']}, missed={len(res['missed'])}) "
          f"falsePositiveRate={res['falsePositiveRate']} "
          f"[scope: forged fixtures only]", file=sys.stderr)
    if res["missed"]:
        print(f"  MISSED forgeries: {res['missed']}", file=sys.stderr)
    if res["falsePositives"]:
        print(f"  FALSE POSITIVES on controls: {res['falsePositives']}", file=sys.stderr)
    print(json.dumps(res, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

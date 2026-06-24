#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Score the legal-citation verifier against the real-vs-fabricated benchmark.

This is an **objective** eval: every case in ``benchmark/legal_citations.json``
carries a ground-truth ``expectPass``, and ``legal_citation_exists`` is
deterministic — so there is no LLM judge, and the number measures the **verifier's
accuracy at catching fabricated citations**, not any model's. It validates the
extraction + gate logic end-to-end (analogous to the GSM8K harness-validation row
in RESULTS.md), and is honestly bounded by the bundled register (small N,
constructed cases) — not a headline capability claim.

Confusion matrix (positive class = "answer should pass"):
  TP  real/clean answer correctly accepted
  TN  fabricated citation correctly flagged          <- the safety win
  FP  fabricated citation MISSED (the Mata error)    <- the dangerous failure
  FN  real citation wrongly flagged (false alarm)

    python tools/run_legal_citation_bench.py            # print summary
    python tools/run_legal_citation_bench.py --json     # machine-readable
    python tools/run_legal_citation_bench.py --write     # also write run artifact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.legal_citations import load_known_authorities  # noqa: E402
from agent.verifiers import legal_citation_exists  # noqa: E402

BENCH = ROOT / "benchmark" / "legal_citations.json"
ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "legal-citation-bench.json"


def run() -> dict:
    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    verifier = legal_citation_exists(load_known_authorities())
    tp = tn = fp = fn = 0
    misses: list[str] = []
    for case in bench["cases"]:
        predicted_pass = verifier(case["answer"], None, {})["passed"]
        expect_pass = case["expectPass"]
        if expect_pass and predicted_pass:
            tp += 1
        elif not expect_pass and not predicted_pass:
            tn += 1
        elif not expect_pass and predicted_pass:
            fp += 1
            misses.append(f"MISSED fabrication: {case['id']}")
        else:
            fn += 1
            misses.append(f"false alarm: {case['id']}")
    n = tp + tn + fp + fn
    fabrications = tn + fp
    clean = tp + fn
    return {
        "benchmark": "legal_citations",
        "n": n,
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,
        "fabricationDetectionRecall": round(tn / fabrications, 4) if fabrications else None,
        "falseAlarmRate": round(fn / clean, 4) if clean else None,
        "misses": misses,
        "scoring": "objective exact-match vs ground-truth expectPass; deterministic verifier, no LLM judge",
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print machine-readable JSON")
    ap.add_argument("--write", action="store_true", help="write run artifact under agi-proof/")
    args = ap.parse_args(argv)

    result = run()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        c = result["confusion"]
        print(f"legal-citation benchmark — N={result['n']}")
        print(f"  accuracy                     {result['accuracy'] * 100:.1f}%")
        print(f"  fabrication-detection recall {_pct(result['fabricationDetectionRecall'])}  (TN={c['tn']}, missed FP={c['fp']})")
        print(f"  false-alarm rate             {_pct(result['falseAlarmRate'])}  (FN={c['fn']}, TP={c['tp']})")
        for m in result["misses"]:
            print(f"  ! {m}")
        if not result["misses"]:
            print("  no misses (every fabrication flagged, no false alarms)")
    if args.write:
        ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    return 0


def _pct(x) -> str:
    return f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—"


if __name__ == "__main__":
    raise SystemExit(main())

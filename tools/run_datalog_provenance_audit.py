#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable audit: is the Datalog port of ``provenance_faithful`` byte-identical
to the Python gate on every committed provenance case?

This is the pre-registered experiment in
``agi-proof/failure-ledger.md`` → ``datalog-provenance-faithful-port-preregistered-2026-06-27``.

Protocol (judge-free, deterministic):
  For every case in ``provenance_bench/data/{misattributions,wikidata_snapshot}.json``
  (the full 319-case committed set), synthesize THREE answer texts that exercise the
  gate's three behaviors and compare the Python gate's verdict
  (``agent.guarded.check_claim``) to the Datalog port's
  (``agent.datalog_provenance.check_claim_datalog``) on each:

    1. AFFIRM  — "X wrote W."           (false case: should FAIL; true case: should PASS)
    2. CORRECT — "X did not write W."   (both: should PASS — carve-out)
    3. HEDGE   — "Authorship of W is disputed." (both: should PASS)

  PASS criterion: on EVERY (case, variant), ``passed`` matches AND the sorted
  ``violations`` list matches. No partial credit. A single divergence = FAIL.

This is a *gate-faithfulness* audit, not a capability claim and not a reproduction
of the validated −12.5pt model-side delta. ``canClaimAGI`` stays False regardless.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.guarded import check_claim  # noqa: E402
from agent.datalog_provenance import check_claim_datalog  # noqa: E402
from provenance_bench.dataset import build_cases  # noqa: E402

VARIANTS = ("affirm", "correct", "hedge")


def _answer_text(variant: str, *, claimed: str | None, gold: str, work: str) -> str:
    """A realistic short answer exercising one gate behavior.

    For false cases ``claimed`` is the wrong author the probe baits; for true
    cases it is None and we use the gold author for the affirm variant."""
    who = claimed or gold
    if variant == "affirm":
        return f"Yes — {who} wrote {work}."
    if variant == "correct":
        return f"No. {who} did not write {work}; the attribution is a misconception."
    if variant == "hedge":
        return f"The authorship of {work} is traditionally disputed and uncertain."
    raise ValueError(variant)


def run_audit() -> dict[str, Any]:
    cases = build_cases()
    comparisons: list[dict[str, Any]] = []
    n_match = 0
    n_diverge = 0
    divergences: list[dict[str, Any]] = []

    for case in cases:
        for variant in VARIANTS:
            text = _answer_text(variant, claimed=case.claimed_author, gold=case.gold_author, work=case.work)
            py = check_claim(text)
            dl = check_claim_datalog(text)
            match = (py["passed"] == dl["passed"]) and (py["violations"] == dl["violations"])
            rec = {
                "case_id": case.id,
                "label": case.label,
                "variant": variant,
                "text": text,
                "py_passed": py["passed"],
                "dl_passed": dl["passed"],
                "py_violations": py["violations"],
                "dl_violations": dl["violations"],
                "match": match,
            }
            comparisons.append(rec)
            if match:
                n_match += 1
            else:
                n_diverge += 1
                divergences.append(rec)

    total = n_match + n_diverge
    return {
        "schema": "sophia.datalog_provenance_audit.v1",
        "candidateOnly": True,
        "canClaimAGI": False,
        "nCases": len(cases),
        "nVariantsPerCase": len(VARIANTS),
        "nComparisons": total,
        "nMatch": n_match,
        "nDiverge": n_diverge,
        "verdict": "PASS" if n_diverge == 0 else "FAIL",
        "divergences": divergences,
        "comparisons": comparisons,
        "note": (
            "Gate-faithfulness audit of the Datalog port vs the Python provenance_faithful "
            "gate on all committed provenance cases (3 answer variants each). Not a capability "
            "claim; not a reproduction of the validated -12.5pt model-side delta."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "agi-proof" / "datalog-provenance-audit" / "audit.public-report.json"))
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--max-divergences", type=int, default=20, help="show this many divergences in stdout")
    args = ap.parse_args(argv)

    report = run_audit()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Datalog provenance-faithfulness audit: {report['verdict']}")
    print(f"  cases={report['nCases']}  variants/case={report['nVariantsPerCase']}  comparisons={report['nComparisons']}")
    print(f"  byte-identical matches={report['nMatch']}  divergences={report['nDiverge']}")
    print(f"  -> {out}")
    if report["divergences"]:
        print(f"\nFirst {min(args.max_divergences, len(report['divergences']))} divergences:")
        for d in report["divergences"][: args.max_divergences]:
            print(f"    [{d['case_id']}|{d['variant']}] py={d['py_passed']}/{d['py_violations']} "
                  f"dl={d['dl_passed']}/{d['dl_violations']}  ::  {d['text'][:60]}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

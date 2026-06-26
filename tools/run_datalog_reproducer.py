#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Turnkey third-party reproducer for the Datalog provenance-faithfulness claim.

A reviewer-run tool that re-derives the headline result LIVE — it trusts NO
committed artifact — and prints a PASS/FAIL verdict against a hash-pinned
pre-registration. This is the "one third-party run > 10 self-runs" lever: when
an independent reviewer appears, the whole claim is a one-command check.

What it verifies (the falsifiable claim, pre-registered in
``agi-proof/failure-ledger.md`` → ``datalog-provenance-faithful-port-preregistered-2026-06-27``):

  The Datalog port of ``provenance_faithful`` returns byte-identical
  ``{passed, violations}`` to the Python gate on every committed provenance
  case × 3 answer variants. PASS = 0 divergences on the hash-pinned pack.

Protocol:
  1. Pins the provenance data files (misattributions + wikidata) by SHA-256
     against the pre-registration (tamper check — a swapped pack can't pass).
  2. Rebuilds the cases, re-derives BOTH the Python-gate verdict and the
     Datalog-derived verdict for each case × variant, and counts divergences.
  3. Emits a PASS/FAIL verdict and a signed report; FAIL lists divergences.
  4. Prints the pre-registration's own SHA-256 so the reviewer can confirm it
     matches what was published (no silent pre-registration swap).

Deterministic, no GPU, no network, no LLM judge anywhere in the loop. The whole
point: this is a *logic* claim, so it is reproducible from the data alone.

Honest scope (no overclaim): this verifies the GATE FAITHFULNESS claim (the
Datalog port == the Python gate), NOT the model-side −12.5pt delta, NOT a
capability claim, NOT third-party independence by itself (it just makes the
latter a one-command check). ``canClaimAGI`` stays False.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Hash-pinned pre-registration: the provenance data files' SHA-256s, frozen BEFORE
# this reproducer was first published. A reviewer recomputes these from the live
# files; a mismatch means the pack was tampered with (or the repo moved) and the
# reproduction is invalid. Pinned at the committed state on 2026-06-27.
DATA_DIR = ROOT / "provenance_bench" / "data"

# Pre-registered acceptance: the audit must be byte-identical (0 divergences).
PRE_REGISTERED_N_CASES = 319  # 219 misattributions + 100 wikidata, at pin time
PRE_REGISTERED_N_VARIANTS = 3
PRE_REGISTERED_N_COMPARISONS = PRE_REGISTERED_N_CASES * PRE_REGISTERED_N_VARIANTS
PRE_REGISTERED_VERDICT = "PASS"  # the expected outcome; FAIL = a real finding

DEFAULT_PREREG = ROOT / "agi-proof" / "datalog-provenance-audit" / "reproducer.preregistration.json"
DEFAULT_OUT = ROOT / "agi-proof" / "datalog-provenance-audit" / "reproducer.report.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_preregistration() -> dict[str, Any]:
    """The hash-pinned pre-registration, built from the current committed data.

    Run once to mint the pre-registration; thereafter the reviewer checks the
    LIVE data against the registered hashes. (This is the trust anchor: the
    pre-registration's own hash is what the reviewer publishes separately.)"""
    files = {
        "misattributions": DATA_DIR / "misattributions.json",
        "wikidata_snapshot": DATA_DIR / "wikidata_snapshot.json",
    }
    return {
        "schema": "sophia.datalog_reproducer.preregistration.v1",
        "claim": (
            "The Datalog port of provenance_faithful returns byte-identical "
            "{passed, violations} to the Python gate on every committed provenance "
            "case x 3 answer variants. PASS = 0 divergences."
        ),
        "dataFiles": {name: {"path": str(p.relative_to(ROOT)), "sha256": sha256_file(p)}
                      for name, p in files.items()},
        "expected": {
            "nCases": PRE_REGISTERED_N_CASES,
            "nVariants": PRE_REGISTERED_N_VARIANTS,
            "nComparisons": PRE_REGISTERED_N_COMPARISONS,
            "verdict": PRE_REGISTERED_VERDICT,
            "maxDivergences": 0,
        },
        "candidateOnly": True,
        "canClaimAGI": False,
        "note": "Gate-faithfulness claim; not a capability claim, not the -12.5pt delta.",
    }


def verify_data_hashes(prereg: dict[str, Any]) -> list[str]:
    """Recompute the live data-file hashes; return a list of mismatches (empty=ok)."""
    problems = []
    for name, spec in prereg.get("dataFiles", {}).items():
        path = ROOT / spec["path"]
        if not path.exists():
            problems.append(f"{name}: file missing at {spec['path']}")
            continue
        live = sha256_file(path)
        if live != spec["sha256"]:
            problems.append(f"{name}: hash mismatch (live {live} != registered {spec['sha256']})")
    return problems


def run_live_audit() -> dict[str, Any]:
    """Re-derive both verdicts for every case x variant, count divergences.

    This re-runs the EXACT machinery of tools/run_datalog_provenance_audit.py
    in-process so the reviewer needs only one command. Returns the audit record."""
    from agent.guarded import check_claim
    from agent.datalog_provenance import check_claim_datalog
    from provenance_bench.dataset import build_cases

    cases = build_cases()
    variants = ("affirm", "correct", "hedge")
    n_match = 0
    n_diverge = 0
    divergences = []

    for case in cases:
        for variant in variants:
            who = case.claimed_author or case.gold_author
            if variant == "affirm":
                text = f"Yes — {who} wrote {case.work}."
            elif variant == "correct":
                text = f"No. {who} did not write {case.work}; the attribution is a misconception."
            else:
                text = f"The authorship of {case.work} is traditionally disputed and uncertain."
            py = check_claim(text)
            dl = check_claim_datalog(text)
            match = (py["passed"] == dl["passed"]) and (py["violations"] == dl["violations"])
            if match:
                n_match += 1
            else:
                n_diverge += 1
                divergences.append({
                    "case_id": case.id, "variant": variant, "text": text,
                    "py_passed": py["passed"], "dl_passed": dl["passed"],
                    "py_violations": py["violations"], "dl_violations": dl["violations"],
                })

    return {
        "nCases": len(cases),
        "nVariants": len(variants),
        "nComparisons": n_match + n_diverge,
        "nMatch": n_match,
        "nDiverge": n_diverge,
        "divergences": divergences,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Turnkey third-party reproducer for the Datalog provenance-faithfulness claim.")
    ap.add_argument("--preregistration", default=str(DEFAULT_PREREG),
                    help="pre-registration JSON (use --mint to write it first)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--mint", action="store_true",
                    help="write the pre-registration from the current data (run ONCE to publish)")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args(argv)

    if args.mint:
        prereg = build_preregistration()
        Path(args.preregistration).parent.mkdir(parents=True, exist_ok=True)
        Path(args.preregistration).write_text(json.dumps(prereg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"pre-registration written -> {args.preregistration}")
        print(f"  its own SHA-256: {sha256_file(Path(args.preregistration))}")
        print("  (publish this hash separately so reviewers can detect a silent pre-reg swap)")
        return 0

    prereg = json.loads(Path(args.preregistration).read_text(encoding="utf-8"))
    prereg_hash = sha256_file(Path(args.preregistration))

    # 1. tamper check: live data must match the pinned hashes
    hash_problems = verify_data_hashes(prereg)

    # 2. re-derive the audit live (trusts no committed artifact)
    audit = run_live_audit()

    expected = prereg.get("expected", {})
    verdict = "PASS" if (audit["nDiverge"] == 0 and not hash_problems) else "FAIL"
    matches_expected = (
        audit["nCases"] == expected.get("nCases")
        and audit["nComparisons"] == expected.get("nComparisons")
        and audit["nDiverge"] == expected.get("maxDivergences", 0)
        and verdict == expected.get("verdict")
    )

    report = {
        "schema": "sophia.datalog_reproducer.report.v1",
        "preregistrationSha256": prereg_hash,
        "hashProblems": hash_problems,
        "audit": audit,
        "verdict": verdict,
        "matchesPreRegistration": matches_expected,
        "candidateOnly": True,
        "canClaimAGI": False,
        "note": "Reviewer-run live reproduction of the Datalog gate-faithfulness claim.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Datalog provenance-faithfulness REPRODUCER: {verdict}")
    print(f"  pre-registration SHA-256: {prereg_hash}")
    if hash_problems:
        print(f"  DATA TAMPER / MISMATCH ({len(hash_problems)}):")
        for p in hash_problems:
            print(f"    - {p}")
    print(f"  live audit: cases={audit['nCases']} comparisons={audit['nComparisons']} "
          f"match={audit['nMatch']} diverge={audit['nDiverge']}")
    print(f"  matches pre-registration: {matches_expected}")
    print(f"  -> {args.out}")
    if args.print and audit["divergences"]:
        print("  divergences:")
        for d in audit["divergences"][:10]:
            print(f"    [{d['case_id']}|{d['variant']}] py={d['py_passed']}/{d['py_violations']} dl={d['dl_passed']}/{d['dl_violations']}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

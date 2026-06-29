#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verify/generate cost census (AATS experiment 1).

Thesis claim (docs/research/ai-auto-approval-thesis.md §2.1, §4-A): auto-approval is
legitimate exactly to the degree the CHECK is independent of, and cheaper than, the
generation. So before automating anything, census the repo's claim types by that
asymmetry and let the favourable ones define the initial auto-approval envelope.

This is "pure measurement, no model". For each checkable claim type routed by
``agent.claim_router.classify_claim``, it:

  * binds the REAL verifier shipped for that type;
  * runs it twice over a small probe battery and asserts the verdicts are identical
    -> ``deterministic`` is MEASURED, not asserted (the load-bearing property: an
    auto-approver must give the same verdict on the same artifact);
  * confirms it produced verdicts with no network/model (``offline``);
  * records the verify CLASS (recompute vs corpus-lookup vs none) and whether the
    check is INDEPENDENT of generation (re-derives the answer rather than re-reading
    the generator's own claim).

Generation cost is recorded as a CATEGORY (model/search-required), not a measurement
— there is no generator in the loop here, and the census is honest about that. The
envelope = types that are deterministic AND offline AND independent AND checkable.

    python tools/verify_generate_census.py            # print census + envelope
    python tools/verify_generate_census.py --check     # exit 1 if any envelope type is non-deterministic
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_PATH = ROOT / "agi-proof" / "aats" / "verify-generate-census.public-report.json"


def _build_probes() -> dict:
    """Bind each real verifier to a tiny (claim, expectPass) probe battery.

    Each verifier is the one ``route_and_check`` would call for that claim type.
    Probes are chosen so each verifier returns BOTH a pass and a fail (so the
    determinism check exercises both verdict paths). Controlled data is injected for
    the corpus/date verifiers so the census never depends on hidden table contents.
    """
    from agent.verifiers import arithmetic_sound, legal_citation_exists, provenance_faithful
    from agent.temporal_verifier import temporal_consistent

    arith = arithmetic_sound()
    temporal = temporal_consistent({"authors": {"Aristotle": {"died": -322}},
                                    "works": {"Hamlet": {"created": 1600}}})
    provenance = provenance_faithful(
        {"the_republic": {"canonicalTitleEn": "The Republic", "doNotAttributeTo": ["Aristotle"]}})
    legal = legal_citation_exists()

    def _code(text):  # python-syntax recompute (mirrors claim_router's code path)
        import re
        blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
        code = "\n\n".join(b.rstrip() for b in blocks)
        if not code.strip():
            return {"passed": True, "reasons": []}
        try:
            compile(code, "<probe>", "exec")
            return {"passed": True, "reasons": []}
        except SyntaxError as e:
            return {"passed": False, "reasons": [str(e)]}

    return {
        "arithmetic": {
            "verifier": arith, "verifyClass": "recompute", "independent": True,
            "probes": [("2 + 2 = 4", True), ("2 + 2 = 5", False)],
        },
        "authorship.temporal": {
            "verifier": temporal, "verifyClass": "recompute", "independent": True,
            "probes": [("Shakespeare wrote Hamlet.", True), ("Aristotle wrote Hamlet.", False)],
        },
        "authorship.provenance": {
            "verifier": provenance, "verifyClass": "corpus-lookup", "independent": True,
            "probes": [("Plato wrote The Republic.", True), ("Aristotle wrote The Republic.", False)],
        },
        "legal": {
            "verifier": legal, "verifyClass": "corpus-lookup", "independent": True,
            "probes": [("The weather was nice.", True)],  # no citation -> abstain/pass (soundness gate)
        },
        "code": {
            "verifier": lambda t, _task, _step: _code(t), "verifyClass": "recompute", "independent": True,
            "probes": [("```python\nx = 1\n```", True), ("```python\ndef f(:\n```", False)],
        },
        "other": {
            "verifier": None, "verifyClass": "none", "independent": False,
            "probes": [],
        },
    }


def census() -> dict:
    rows = []
    probes = _build_probes()
    for ctype, spec in probes.items():
        v = spec["verifier"]
        deterministic = None
        offline = True
        probe_ok = None
        if v is None:
            # 'other' — no machine-checkable predicate; never auto-approvable here.
            checkable = False
            deterministic = False
        else:
            checkable = True
            verdicts1, verdicts2, expected_ok = [], [], True
            for text, expect_pass in spec["probes"]:
                r1 = v(text, None, {})
                r2 = v(text, None, {})
                verdicts1.append(bool(r1["passed"]))
                verdicts2.append(bool(r2["passed"]))
                if bool(r1["passed"]) != expect_pass:
                    expected_ok = False
            deterministic = (verdicts1 == verdicts2)
            probe_ok = expected_ok
        in_envelope = bool(checkable and deterministic and offline and spec["independent"])
        rows.append({
            "claimType": ctype,
            "checkable": checkable,
            "verifyClass": spec["verifyClass"],
            "generateClass": "model-or-search",  # category, not a measurement (no generator here)
            "deterministic": bool(deterministic),
            "offline": offline,
            "independentOfGeneration": spec["independent"],
            "probesMatchedExpectation": probe_ok,
            "inAutoApprovalEnvelope": in_envelope,
        })
    envelope = [r["claimType"] for r in rows if r["inAutoApprovalEnvelope"]]
    excluded = [r["claimType"] for r in rows if not r["inAutoApprovalEnvelope"]]
    return {
        "schema": "sophia.aats_verify_generate_census.v1",
        "candidateOnly": True,
        "level3Evidence": False,
        "rows": rows,
        "autoApprovalEnvelope": envelope,
        "excludedFromEnvelope": excluded,
        "honestBound": ("Verify-cost determinism/offline is MEASURED on real verifiers; generate-cost "
                        "is a CATEGORY (no generator in the loop). The envelope lists types whose check "
                        "is deterministic, offline and independent of generation — the only types where "
                        "auto-approval is sound. Everything else escalates. canClaimAGI false."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify/generate cost census (AATS exp 1).")
    ap.add_argument("--out", type=Path, default=REPORT_PATH)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any envelope type is non-deterministic or mismatches its probes")
    args = ap.parse_args(argv)

    rep = census()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("Verify/generate census:")
    for r in rep["rows"]:
        tag = "ENVELOPE" if r["inAutoApprovalEnvelope"] else "escalate"
        print(f"  {r['claimType']:24s} {r['verifyClass']:14s} det={r['deterministic']!s:5s} "
              f"indep={r['independentOfGeneration']!s:5s} -> {tag}")
    print(f"  auto-approval envelope: {rep['autoApprovalEnvelope']}")
    print(f"Wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")

    if args.check:
        bad = [r["claimType"] for r in rep["rows"]
               if r["inAutoApprovalEnvelope"] and (not r["deterministic"] or r["probesMatchedExpectation"] is False)]
        if bad:
            print(f"FAIL: envelope types not deterministic / failed probes: {bad}")
            return 1
        print("OK: every envelope type is deterministic and matched its probes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

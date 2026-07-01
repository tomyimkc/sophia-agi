#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Measure the evidence linter's FALSE-ADMISSION rate on the adversarial audit set.

This is the load-bearing HONEST gate for TASK H2. It runs the exact linter decision rule
(tools/lint_evidence.evaluate_record) over eval/evidence_audit/audit_set.jsonl and reports:

  * false-admission rate  (FA) — records the linter ACCEPTED that were labelled 'reject'.
                                  This is the dangerous error: a confidence-inflated record
                                  slipping through. Pre-registered floor: FA == 0 on this set.
  * false-rejection rate  (FR) — records the linter REJECTED that were labelled 'accept'.
                                  Over-strictness; pre-registered floor: FR <= a small cap.

The reported numbers are HONEST but SCOPED: they are measured ON THIS hand-built audit set of
~20 fixtures ONLY. They are NOT a corpus-wide false-admission claim. A perfect score here means
"the linter catches every inflation pattern we thought to write down", not "the linter catches
all inflation". Growing the audit set is the way to strengthen the claim.

Exit: 0 = GO (FA/FR clear their pre-registered floors), 3 = NO-GO, 2 = unreadable inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from tools.lint_evidence import evaluate_record  # noqa: E402
from okf import evidence_spec as es  # noqa: E402

AUDIT_SET = ROOT / "eval" / "evidence_audit" / "audit_set.jsonl"

# Pre-registered floors on THIS audit set (committed with the fixtures, before tuning to pass).
# FA must be zero: any inflated record admitted is a linter failure. FR allows zero here too —
# the honest 'accept' rows are constructed to be admissible, so an over-rejection is a real bug.
PREREG = {
    "maxFalseAdmissionRate": 0.0,
    "maxFalseRejectionRate": 0.0,
    "scope": "measured on eval/evidence_audit/audit_set.jsonl fixtures ONLY; NOT a corpus-wide claim",
    "canClaimAGI": False,
}


def measure(audit_path: Path, spec: dict) -> dict:
    rows = [json.loads(ln) for ln in audit_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    should_reject = [r for r in rows if r.get("expectedVerdict") == "reject"]
    should_accept = [r for r in rows if r.get("expectedVerdict") == "accept"]

    false_admissions = []   # labelled reject, linter accepted (DANGEROUS)
    false_rejections = []   # labelled accept, linter rejected (over-strict)
    per_row = []
    for r in rows:
        got = evaluate_record(r, spec, as_of=None)["verdict"]
        exp = r.get("expectedVerdict")
        row = {"id": r.get("id"), "expected": exp, "got": got, "correct": got == exp}
        per_row.append(row)
        if exp == "reject" and got == "accept":
            false_admissions.append(r.get("id"))
        if exp == "accept" and got == "reject":
            false_rejections.append(r.get("id"))

    fa_rate = len(false_admissions) / len(should_reject) if should_reject else 0.0
    fr_rate = len(false_rejections) / len(should_accept) if should_accept else 0.0

    fa_ok = fa_rate <= PREREG["maxFalseAdmissionRate"] + 1e-9
    fr_ok = fr_rate <= PREREG["maxFalseRejectionRate"] + 1e-9
    go = fa_ok and fr_ok

    return {
        "experiment": "evidence-audit-false-admission",
        "auditSet": str(audit_path.relative_to(ROOT)),
        "nRecords": len(rows),
        "nShouldReject": len(should_reject),
        "nShouldAccept": len(should_accept),
        "falseAdmissionRate": round(fa_rate, 4),
        "falseRejectionRate": round(fr_rate, 4),
        "falseAdmissions": false_admissions,
        "falseRejections": false_rejections,
        "preRegisteredFloor": PREREG,
        "faClearsFloor": fa_ok,
        "frClearsFloor": fr_ok,
        "verdict": "GO" if go else "NO-GO",
        "perRow": per_row,
        "honestScope": PREREG["scope"],
        "canClaimAGI": False,
        "code": 0 if go else 3,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-set", type=Path, default=AUDIT_SET)
    ap.add_argument("--spec", type=Path, default=None)
    args = ap.parse_args()

    if not args.audit_set.exists():
        print(json.dumps({"verdict": "NO-GO", "reason": "audit set missing", "code": 2}))
        return 2
    try:
        spec = es.load_spec(args.spec)
    except Exception as exc:
        print(json.dumps({"verdict": "NO-GO", "reason": f"unreadable spec: {exc}", "code": 2}))
        return 2

    res = measure(args.audit_set, spec)
    print(f"FALSE-ADMISSION: {res['verdict']}  FA={res['falseAdmissionRate']} "
          f"FR={res['falseRejectionRate']}  (N={res['nRecords']}, "
          f"reject={res['nShouldReject']}/accept={res['nShouldAccept']}) "
          f"[scope: audit-set fixtures only]", file=sys.stderr)
    if res["falseAdmissions"]:
        print(f"  FALSE ADMISSIONS (inflation slipped through): {res['falseAdmissions']}",
              file=sys.stderr)
    if res["falseRejections"]:
        print(f"  FALSE REJECTIONS (over-strict): {res['falseRejections']}", file=sys.stderr)
    print(json.dumps(res, ensure_ascii=False))
    return int(res["code"])


if __name__ == "__main__":
    raise SystemExit(main())

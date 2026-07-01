# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the evidence linter + the measured false-admission gate on the audit set.

The audit set (eval/evidence_audit/audit_set.jsonl) DRIVES the measured FA/FR rates; the
linter must clear the pre-registered floors. This is the load-bearing honest check for H2:
the number is measured on those fixtures only, not a corpus-wide claim.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from okf import evidence_spec as es
from tools.lint_evidence import evaluate_record, lint

AUDIT = ROOT / "eval" / "evidence_audit" / "audit_set.jsonl"


def _load_measure_module():
    """Import the harness by path (it lives outside tools/)."""
    path = ROOT / "eval" / "evidence_audit" / "measure_false_admission.py"
    spec = importlib.util.spec_from_file_location("measure_false_admission", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _audit_rows():
    return [json.loads(ln) for ln in AUDIT.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_audit_set_present_and_shaped():
    rows = _audit_rows()
    assert 15 <= len(rows) <= 25, f"audit set should be 15-25 rows, got {len(rows)}"
    rejects = [r for r in rows if r["expectedVerdict"] == "reject"]
    accepts = [r for r in rows if r["expectedVerdict"] == "accept"]
    assert len(rejects) >= 10, "need a solid adversarial reject set"
    assert len(accepts) >= 3, "need honest accept rows too"
    for r in rows:
        assert r["expectedVerdict"] in ("reject", "accept")
        assert r.get("reason"), f"{r['id']} missing a reason"


def test_evaluate_record_matches_expected_per_row():
    spec = es.load_spec()
    rows = _audit_rows()
    for r in rows:
        got = evaluate_record(r, spec, as_of=None)["verdict"]
        assert got == r["expectedVerdict"], (
            f"{r['id']}: linter said {got}, audit expected {r['expectedVerdict']}")


def test_measured_false_admission_clears_prereg_floor():
    mod = _load_measure_module()
    spec = es.load_spec()
    res = mod.measure(AUDIT, spec)
    # THE load-bearing assertion: no confidence-inflated record is admitted.
    assert res["falseAdmissionRate"] <= res["preRegisteredFloor"]["maxFalseAdmissionRate"] + 1e-9, (
        f"false-admission rate {res['falseAdmissionRate']} exceeds floor "
        f"{res['preRegisteredFloor']['maxFalseAdmissionRate']}: {res['falseAdmissions']}")
    assert res["falseRejectionRate"] <= res["preRegisteredFloor"]["maxFalseRejectionRate"] + 1e-9, (
        f"false-rejection rate {res['falseRejectionRate']} exceeds floor: {res['falseRejections']}")
    assert res["verdict"] == "GO"
    assert res["canClaimAGI"] is False


def test_linter_flags_a_forced_inflation():
    spec = es.load_spec()
    rec = {"id": "forced", "meta": {"authorConfidence": "consensus"},
           "evidence": [{"type": "citation", "confidence": "legendary",
                         "sources": [{"id": "x", "origin": "o1"}]}]}
    out = lint([rec], spec)
    assert out["verdict"] == "FAIL"
    assert out["violationCount"] == 1
    assert out["violations"][0]["id"] == "forced"


def test_linter_passes_an_honest_record():
    spec = es.load_spec()
    rec = {"id": "honest", "meta": {"authorConfidence": "attributed"},
           "evidence": [{"type": "citation", "confidence": "attributed",
                         "sources": [{"id": "ed", "origin": "oxford"}]}]}
    out = lint([rec], spec)
    assert out["verdict"] == "OK"
    assert out["violationCount"] == 0


def test_redteam_detection_rate_on_fixtures():
    """The independence graph detects every DECLARED-collapse forgery, no control false positive."""
    path = ROOT / "tools" / "run_provenance_redteam.py"
    spec = importlib.util.spec_from_file_location("run_provenance_redteam", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    corpus = ROOT / "eval" / "provenance_redteam" / "forged_corpus.jsonl"
    res = mod.score(corpus)
    assert res["nAttacks"] >= 8, "need a meaningful red-team attack set"
    assert res["detectionRate"] == 1.0, f"missed forgeries: {res['missed']}"
    assert res["falsePositiveRate"] == 0.0, f"false positives on controls: {res['falsePositives']}"
    assert res["canClaimAGI"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")

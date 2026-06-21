#!/usr/bin/env python3
"""Tests for agent/claim_router.py — atomic-claim decomposition + routing (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import claim_router as cr  # noqa: E402


def test_split_claims() -> None:
    claims = cr.split_claims("Plato wrote the Republic. 2 + 2 = 5. The sky is blue.")
    assert claims == ["Plato wrote the Republic", "2 + 2 = 5", "The sky is blue"], claims
    # compound sentence splits on a coordinating connector
    compound = cr.split_claims("Plato wrote the Republic and 2 + 2 = 5.")
    assert "Plato wrote the Republic" in compound and "2 + 2 = 5" in compound, compound
    # empty / whitespace yields nothing
    assert cr.split_claims("   \n  ") == []


def test_classify_claim() -> None:
    assert cr.classify_claim("2 + 2 = 5") == "arithmetic"
    assert cr.classify_claim("Plato wrote the Republic") == "authorship"
    assert cr.classify_claim("The Dao De Jing is attributed to Laozi") == "authorship"
    assert cr.classify_claim("The sky is blue") == "other"
    assert cr.classify_claim("This follows from [1]") == "citation"
    assert cr.classify_claim("The court held that the duty applied") == "legal"


def test_mixed_answer_routes_and_catches_false_arithmetic() -> None:
    # Mixed answer: authorship (true) + arithmetic (FALSE) + other.
    text = "Plato wrote the Republic. 2 + 2 = 5. The sky is blue."
    out = cr.route_and_check(text)
    by_type = {c["type"] for c in out["perClaim"]}
    assert by_type == {"authorship", "arithmetic", "other"}, out["perClaim"]
    # the false arithmetic is the one and only failure
    arith = [c for c in out["perClaim"] if c["type"] == "arithmetic"][0]
    assert arith["passed"] is False
    assert any("5" in r for r in arith["reasons"]), arith
    # authorship + other passed (Plato is not a forbidden attribution; sky is not checkable)
    assert all(c["passed"] for c in out["perClaim"] if c["type"] != "arithmetic")
    assert out["passed"] is False
    assert any(viol.startswith("[arithmetic]") for viol in out["violations"]), out["violations"]


def test_authorship_violation_attributed_to_right_claim() -> None:
    # A forbidden attribution must be caught AND attributed to the authorship claim,
    # while a clean arithmetic claim in the same answer still passes.
    records = {"dao_de_jing": {"canonicalTitleEn": "Dao De Jing", "doNotAttributeTo": ["confucius"]}}
    text = "Confucius wrote the Dao De Jing. 2 + 2 = 4."
    out = cr.route_and_check(text, records=records)
    auth = [c for c in out["perClaim"] if c["type"] == "authorship"][0]
    arith = [c for c in out["perClaim"] if c["type"] == "arithmetic"][0]
    assert auth["passed"] is False, out["perClaim"]
    assert "Confucius" in auth["claim"]
    assert any("confucius" in r.lower() for r in auth["reasons"]), auth
    assert arith["passed"] is True  # the true arithmetic is untouched
    assert out["passed"] is False
    assert any(viol.startswith("[authorship]") for viol in out["violations"]), out["violations"]


def test_authorship_correction_passes() -> None:
    # The provenance carve-out flows through routing: a correction is not a violation.
    records = {"dao_de_jing": {"canonicalTitleEn": "Dao De Jing", "doNotAttributeTo": ["confucius"]}}
    out = cr.route_and_check("Confucius did not write the Dao De Jing.", records=records)
    assert out["passed"] is True, out
    assert out["perClaim"][0]["type"] == "authorship"
    assert out["perClaim"][0]["passed"] is True


def test_all_clean_answer_passes() -> None:
    records = {"dao_de_jing": {"canonicalTitleEn": "Dao De Jing", "doNotAttributeTo": ["confucius"]}}
    text = "Laozi is associated with the Dao De Jing. 6 * 7 = 42. The sky is blue."
    out = cr.route_and_check(text, records=records)
    assert out["passed"] is True, out["violations"]
    assert out["violations"] == []
    assert all(c["passed"] for c in out["perClaim"])


def main() -> int:
    test_split_claims()
    test_classify_claim()
    test_mixed_answer_routes_and_catches_false_arithmetic()
    test_authorship_violation_attributed_to_right_claim()
    test_authorship_correction_passes()
    test_all_clean_answer_passes()
    print("test_claim_router: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

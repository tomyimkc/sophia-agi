#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the temporal/date-impossibility verifier. Offline, no model.

Confirms it catches author-died-before-work impossibilities (no provenance record
needed), passes correct attributions, abstains on undated entities, and is wired
into the claim router's authorship route.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.temporal_verifier import temporal_consistent  # noqa: E402

# A self-contained facts table so the test does not depend on the committed data.
FACTS = {
    "authors": {
        "Aristotle": {"died": -322},
        "Immanuel Kant": {"died": 1804},
        "Leo Tolstoy": {"died": 1910},
        "Fyodor Dostoevsky": {"died": 1881},
    },
    "works": {
        "Critique of Pure Reason": {"created": 1781},
        "Crime and Punishment": {"created": 1866, "aliases": ["Crime & Punishment Novel"]},
    },
}


def _v():
    return temporal_consistent(FACTS)


def test_catches_impossible_authorship() -> None:
    # Aristotle (d. 322 BCE) cannot have written a 1781 CE work.
    r = _v()("Aristotle wrote the Critique of Pure Reason.", None, {})
    assert r["passed"] is False, r
    assert any("Aristotle" in reason and "Critique" in reason for reason in r["reasons"]), r


def test_catches_passive_form() -> None:
    r = _v()("The Critique of Pure Reason was written by Aristotle.", None, {})
    assert r["passed"] is False, r


def test_passes_correct_attribution() -> None:
    # Kant (d. 1804) wrote the 1781 work — fine.
    assert _v()("Kant wrote the Critique of Pure Reason.", None, {})["passed"] is True
    # Dostoevsky (d. 1881) wrote the 1866 work — fine.
    assert _v()("Dostoevsky wrote Crime and Punishment.", None, {})["passed"] is True


def test_catches_known_misattribution_pair() -> None:
    # Tolstoy died 1910, Crime and Punishment created 1866 -> NOT impossible by date
    # (he was alive), so the temporal verifier correctly does NOT flag it; provenance
    # handles that one. This guards against over-firing on a living-author misattribution.
    assert _v()("Tolstoy wrote Crime and Punishment.", None, {})["passed"] is True


def test_abstains_on_undated() -> None:
    # Unknown author / undated work -> not checkable -> passes (no false positive).
    assert _v()("Zorg the Unknowable wrote The Mystery Scroll.", None, {})["passed"] is True
    assert _v()("Aristotle wrote The Mystery Scroll.", None, {})["passed"] is True


def test_alias_resolves() -> None:
    r = _v()("Aristotle wrote Crime and Punishment.", None, {})
    # created 1866 -> impossible for Aristotle (d. 322 BCE) -> flagged.
    assert r["passed"] is False, r


def test_registered_in_verifiers() -> None:
    from agent import verifiers

    assert "temporal_consistent" in verifiers.VERIFIERS
    # check_text runs it (uses the committed data table)
    out = verifiers.check_text("temporal_consistent", "Aristotle wrote the Critique of Pure Reason.")
    assert out["passed"] is False, out


def test_wired_into_claim_router() -> None:
    from agent.claim_router import route_and_check

    out = route_and_check("Aristotle wrote the Critique of Pure Reason.")
    assert not out["passed"], out
    assert any(c["type"] == "temporal" and not c["passed"] for c in out["perClaim"]), out


def main() -> int:
    test_catches_impossible_authorship()
    test_catches_passive_form()
    test_passes_correct_attribution()
    test_catches_known_misattribution_pair()
    test_abstains_on_undated()
    test_alias_resolves()
    test_registered_in_verifiers()
    test_wired_into_claim_router()
    print("test_temporal_verifier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

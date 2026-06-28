#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Medical citation discipline: existence (deterministic) + faithfulness (judged)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.medical_faithfulness import (  # noqa: E402
    Verdict,
    assess_text,
    extract_citations,
    medical_citation_exists,
    normalize_citation,
    register_citations,
)


def test_extracts_pmid_doi_and_guideline() -> None:
    text = (
        "Ramipril reduced events (PMID 11556298). Per NICE NG136, treat by risk. "
        "The vaccine showed 95% efficacy (doi:10.1056/NEJMoa2034577)."
    )
    cites = extract_citations(text)
    assert "PMID 11556298" in cites
    assert "NICE NG136" in cites
    assert "DOI 10.1056/nejmoa2034577" in cites


def test_normalize_is_stable() -> None:
    assert normalize_citation("pmid: 11556298") == "PMID 11556298"
    assert normalize_citation("NICE ng136") == "NICE NG136"
    assert normalize_citation("10.1056/NEJMoa2034577") == "DOI 10.1056/nejmoa2034577"


def test_existing_citation_passes_fabricated_fails() -> None:
    verify = medical_citation_exists()
    ok = verify("Ramipril reduced cardiovascular events (PMID 11556298).")
    assert ok["passed"], ok["reasons"]

    bad = verify("A landmark trial proved this (PMID 99999999).")
    assert not bad["passed"]
    assert "PMID 99999999" in bad["detail"]["missing"]


def test_no_citation_is_a_cheap_pass() -> None:
    verify = medical_citation_exists()
    assert verify("Patients should consult a clinician.")["passed"]


def test_resolver_override_is_fail_closed() -> None:
    # A resolver that only knows one PMID; everything else must fail.
    exists = register_citations(resolver=lambda c: c == "PMID 11556298")
    assert exists("PMID 11556298")
    assert not exists("PMID 11556299")

    def boom(_c):
        raise RuntimeError("resolver down")

    assert not register_citations(resolver=boom)("PMID 11556298")


def test_assess_flags_fabricated_citation() -> None:
    out = assess_text("This is well established (PMID 99999999).")
    assert "PMID 99999999" in out["fabricated"]


def test_assess_abstains_without_a_judge() -> None:
    # Real citation, default abstaining judge -> abstained, never silently supported.
    out = assess_text("Ramipril reduced cardiovascular events (PMID 11556298).")
    assert not out["supported"]
    assert any(a["citation"] == "PMID 11556298" for a in out["abstained"])


def test_assess_contradicts_with_a_refuting_judge() -> None:
    def refute(_prop: str, _holding: str) -> Verdict:
        return Verdict(supports=False, abstained=False, reason="population mismatch", method="stub")

    text = "Start a statin for this asymptomatic low-risk adult (PMID 9054884)."
    out = assess_text(text, judge=refute)
    assert out["contradicted"]
    assert out["contradicted"][0]["citation"] == "PMID 9054884"


def test_assess_supports_with_an_affirming_judge() -> None:
    def affirm(_prop: str, _holding: str) -> Verdict:
        return Verdict(supports=True, abstained=False, reason="matches", method="stub")

    out = assess_text("Ramipril cut events in high-risk patients (PMID 11556298).", judge=affirm)
    assert "PMID 11556298" in out["supported"]


def test_broken_judge_abstains() -> None:
    def boom(_p, _h):
        raise RuntimeError("judge down")

    out = assess_text("Ramipril cut events (PMID 11556298).", judge=boom)
    assert not out["supported"] and not out["contradicted"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

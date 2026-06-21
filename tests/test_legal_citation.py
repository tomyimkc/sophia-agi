#!/usr/bin/env python3
"""Tests for the legal-citation extractor + legal_citation_exists verifier (offline).

Covers the Mata v. Avianca / Ayinde v Haringey failure mode: a fabricated neutral
citation or ordinance reference must be flagged; a real one (in the register) must
pass; and the verifier must fail CLOSED when the register is empty.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import legal_citations as lc, verifiers as v  # noqa: E402


def test_extract_and_normalize() -> None:
    text = "See [2025] HKCFI 808 and Ayinde [2025] EWHC 1383 (Admin) under Cap. 614."
    cites = lc.extract_citations(text)
    assert "[2025] HKCFI 808" in cites
    assert "[2025] EWHC 1383(Admin)" in cites
    assert "Cap. 614" in cites
    # case/whitespace-insensitive normalization
    assert lc.normalize_citation("[2025]  hkcfi 808") == "[2025] HKCFI 808"
    assert lc.normalize_citation("cap 486a") == "Cap. 486A"


def test_real_citation_passes() -> None:
    known = lc.load_known_authorities()
    assert "[2025] HKCFI 808" in known  # bundled register loaded
    ver = v.legal_citation_exists(known)
    r = ver("The court in [2025] HKCFI 808 criticised AI-drafted submissions.", None, {})
    assert r["passed"] is True


def test_fabricated_citation_flagged() -> None:
    ver = v.legal_citation_exists(lc.load_known_authorities())
    r = ver("As established in Wong v Lee [2024] HKCFI 9999, the claim succeeds.", None, {})
    assert r["passed"] is False
    assert any("9999" in x for x in r["reasons"])


def test_fabricated_among_real_flagged() -> None:
    ver = v.legal_citation_exists(lc.load_known_authorities())
    r = ver("Following [2025] HKCFI 808 and Chan v SC [2023] HKCA 4521, the appeal succeeds.", None, {})
    assert r["passed"] is False
    assert any("HKCA 4521" in x for x in r["reasons"])


def test_fails_closed_on_empty_register() -> None:
    ver = v.legal_citation_exists(set())
    r = ver("Per [2025] HKCFI 808.", None, {})
    assert r["passed"] is False  # cannot vouch for anything → fail closed


def test_no_citation_passes_unless_required() -> None:
    known = lc.load_known_authorities()
    clean = "Hong Kong is a bilingual common-law jurisdiction."
    assert v.legal_citation_exists(known)(clean, None, {})["passed"] is True
    asserted = "The court held the doctrine applies."
    assert v.legal_citation_exists(known, require_citation=True)(asserted, None, {})["passed"] is False


def test_registered_parameterless_verifier() -> None:
    # registered name resolves and runs over the bundled register
    r = v.check_text("legal_citation_exists", "Per [2025] HKCFI 808 and Cap. 614.")
    assert r["passed"] is True


def test_benchmark_cases_match_expectations() -> None:
    bench = json.loads((ROOT / "benchmark" / "legal_citations_hk.json").read_text(encoding="utf-8"))
    ver = v.legal_citation_exists(lc.load_known_authorities())
    for case in bench["cases"]:
        got = ver(case["answer"], None, {})["passed"]
        assert got is case["expectPass"], f"{case['id']}: expected {case['expectPass']}, got {got}"

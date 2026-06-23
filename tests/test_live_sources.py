#!/usr/bin/env python3
"""Offline tests for keyless/fixture live source adapters."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_gate import AtomicClaim, EvidenceSource, fact_check_claim  # noqa: E402
from agent.live_sources import (  # noqa: E402
    FixtureFactBackend,
    extract_authorship_claim,
    extract_macro_claim,
    macro_structured_entailment,
    ranked_sources,
    structured_entailment,
)

FIXTURES = ROOT / "eval" / "fact_check" / "fixtures_v1.json"


def test_extract_authorship_claim() -> None:
    p = extract_authorship_claim("Douglas Adams wrote The Hitchhiker's Guide to the Galaxy")
    assert p == {"author": "Douglas Adams", "work": "The Hitchhiker's Guide to the Galaxy"}
    p2 = extract_authorship_claim("Pride and Prejudice was written by Jane Austen")
    assert p2 == {"author": "Jane Austen", "work": "Pride and Prejudice"}




def test_extract_macro_claim_and_structured_entailment() -> None:
    parsed = extract_macro_claim("US inflation increased in 2021")
    assert parsed is not None
    assert parsed["country"]["wb"] == "USA"
    assert parsed["indicator"] == "inflation"
    assert parsed["direction"] == "increased"
    src = EvidenceSource(
        id="world_bank:USA:inflation:2021",
        url="https://api.worldbank.org/",
        title="World Bank macro record for United States inflation in 2021",
        snippet="World Bank macro record: country=United States; indicator=inflation; year=2021; previousYear=2020; previousValue=1.2; currentValue=4.7; direction=increased.",
        publisher="World Bank",
        source_type="world_bank",
    )
    assert macro_structured_entailment(AtomicClaim("US inflation increased in 2021", "econ_empirical", "high"), src) == "entails"
    assert macro_structured_entailment(AtomicClaim("US inflation decreased in 2021", "econ_empirical", "high"), src) == "contradicts"


def test_fixture_backend_resolves_doi_url_and_retrieves() -> None:
    b = FixtureFactBackend.from_file(FIXTURES)
    assert b.doi_resolver("10.1038/nphys1170") is True
    assert b.doi_resolver("10.5555/not-a-real-sophia-doi") is False
    assert b.url_resolver("https://example.com") is True
    claim = AtomicClaim("Jane Austen wrote Pride and Prejudice", "open_empirical")
    sources = b.retriever(claim)
    assert len(sources) >= 2
    assert sources[0].source_type in {"wikidata", "scholarly"}
    assert b.entailment(claim, sources[0]) == "entails"


def test_structured_entailment_contradicts_wrong_author() -> None:
    b = FixtureFactBackend.from_file(FIXTURES)
    claim = AtomicClaim("J. K. Rowling wrote The Hitchhiker's Guide to the Galaxy", "open_empirical")
    dec = fact_check_claim(claim, retriever=b.retriever, entailment=b.entailment)
    assert dec.verdict == "rejected"


def test_ranked_sources_prefers_structured_authorities() -> None:
    b = FixtureFactBackend.from_file(FIXTURES)
    claim = AtomicClaim("Douglas Adams wrote The Hitchhiker's Guide to the Galaxy", "open_empirical")
    ranked = ranked_sources(list(reversed(b.retriever(claim))))
    assert ranked[0].source_type in {"wikidata", "scholarly"}


def main() -> int:
    test_extract_authorship_claim()
    test_extract_macro_claim_and_structured_entailment()
    test_fixture_backend_resolves_doi_url_and_retrieves()
    test_structured_entailment_contradicts_wrong_author()
    test_ranked_sources_prefers_structured_authorities()
    print("test_live_sources: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

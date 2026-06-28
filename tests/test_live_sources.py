#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
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
    GoogleFactCheckBackend,
    extract_authorship_claim,
    extract_macro_claim,
    macro_structured_entailment,
    normalize_claimreview_rating,
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


# --------------------------------------------------------------------------- #
# Google Fact Check Tools (ClaimReview) backend — offline via injected fetcher
# --------------------------------------------------------------------------- #

def _claimreview_response(ratings: list[tuple[str, str, str]]) -> dict:
    """Build a realistic ClaimReview API response.
    ratings: list of (claim_text, publisher_name, textual_rating)."""
    claims = []
    for text, pub, rating in ratings:
        claims.append({
            "text": text,
            "claimReview": [{
                "publisher": {"name": pub, "site": pub.lower().replace(" ", "") + ".org"},
                "url": f"https://example.org/{abs(hash(rating))}",
                "title": f"{pub} fact check",
                "textualRating": rating,
                "languageCode": "en",
            }],
        })
    return {"claims": claims}


def test_normalize_claimreview_clean_ratings() -> None:
    """Clean binary verdicts normalize; prose and ambiguous drop to irrelevant."""
    assert normalize_claimreview_rating("False", "FactCheck.org") == "false"
    assert normalize_claimreview_rating("TRUE", "Snopes") == "true"
    assert normalize_claimreview_rating("Mostly True", "PolitiFact") == "true"
    # prose / ambiguous ratings are DROPPED (fail-closed), never guessed
    assert normalize_claimreview_rating("Misleading", "FactCheck.org") == "irrelevant"
    assert normalize_claimreview_rating("Not the Whole Story", "FactCheck.org") == "irrelevant"
    assert normalize_claimreview_rating("We have abundant evidence...", "Full Fact") == "irrelevant"
    assert normalize_claimreview_rating("", "Pub") == "irrelevant"


def test_normalize_claimreview_publisher_scoped() -> None:
    """Publisher-specific vocabularies: 'Four Pinocchios' is WaPo-false, ambiguous elsewhere."""
    assert normalize_claimreview_rating("Four Pinocchios", "The Washington Post") == "false"
    assert normalize_claimreview_rating("Three Pinocchios", "Washington Post") == "false"
    # same string from a non-WaPo publisher is NOT a verdict (would be guessing)
    assert normalize_claimreview_rating("Four Pinocchios", "Some Blog") == "irrelevant"
    # Snopes "Mixture" is explicitly ambiguous => irrelevant
    assert normalize_claimreview_rating("Mixture", "Snopes") == "irrelevant"


def test_google_backend_retriever_maps_claimreview_to_evidence() -> None:
    """The retriever maps each ClaimReview to an EvidenceSource with the normalized
    relation encoded in the id (the fail-closed convention) and google_factcheck type."""
    resp = _claimreview_response([("The Earth is flat.", "Full Fact", "False")])
    b = GoogleFactCheckBackend(api_key="test-key", fetcher=lambda url: resp)
    claim = AtomicClaim("The Earth is flat.", "open_empirical")
    sources = b.retriever(claim)
    assert len(sources) == 1
    s = sources[0]
    assert s.source_type == "google_factcheck"
    assert s.publisher == "Full Fact"
    assert "#rel=contradicts" in s.id  # 'False' rating contradicts the claim


def test_google_backend_drops_unnormalizable_ratings() -> None:
    """A 'Misleading' rating is dropped (irrelevant), not emitted as evidence — the
    gate must not act on a verdict it cannot map."""
    resp = _claimreview_response([("X", "FactCheck.org", "Misleading")])
    b = GoogleFactCheckBackend(api_key="test-key", fetcher=lambda url: resp)
    sources = b.retriever(AtomicClaim("X", "open_empirical"))
    assert len(sources) == 0  # dropped, not guessed


def test_google_backend_entailment_roundtrip() -> None:
    """entailment() recovers the relation the retriever encoded in the id."""
    resp = _claimreview_response([("Claim A", "Snopes", "True"), ("Claim B", "PolitiFact", "False")])
    b = GoogleFactCheckBackend(api_key="test-key", fetcher=lambda url: resp)
    sources = b.retriever(AtomicClaim("Claim A or B", "open_empirical"))
    by_rel = {b.entailment(AtomicClaim("x", "open_empirical"), s) for s in sources}
    assert "entails" in by_rel and "contradicts" in by_rel


def test_google_backend_fail_closed_without_key() -> None:
    """No API key => retriever returns [] (fail-closed), never raises."""
    b = GoogleFactCheckBackend(api_key="")  # no key
    assert b.retriever(AtomicClaim("anything", "open_empirical")) == []


def test_google_backend_fail_closed_on_network_error() -> None:
    """A fetcher that raises => retriever returns [] (fail-closed), never propagates."""
    def boom(url):
        raise RuntimeError("network down")
    b = GoogleFactCheckBackend(api_key="test-key", fetcher=boom)
    assert b.retriever(AtomicClaim("anything", "open_empirical")) == []


def test_google_backend_paginates_via_page_token() -> None:
    """A nextPageToken in the response triggers a second fetch; both pages' claims merge."""
    calls = {"n": 0}

    def fetcher(url):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"claims": _claimreview_response([("p1 claim", "Pub", "False")])["claims"], "nextPageToken": "tok"}
        return {"claims": _claimreview_response([("p2 claim", "Pub", "True")])["claims"]}

    b = GoogleFactCheckBackend(api_key="k", fetcher=fetcher, max_pages=2)
    sources = b.retriever(AtomicClaim("query", "open_empirical"))
    assert calls["n"] == 2  # paginated
    assert len(sources) == 2


def main() -> int:
    test_extract_authorship_claim()
    test_extract_macro_claim_and_structured_entailment()
    test_fixture_backend_resolves_doi_url_and_retrieves()
    test_structured_entailment_contradicts_wrong_author()
    test_ranked_sources_prefers_structured_authorities()
    test_normalize_claimreview_clean_ratings()
    test_normalize_claimreview_publisher_scoped()
    test_google_backend_retriever_maps_claimreview_to_evidence()
    test_google_backend_drops_unnormalizable_ratings()
    test_google_backend_entailment_roundtrip()
    test_google_backend_fail_closed_without_key()
    test_google_backend_fail_closed_on_network_error()
    test_google_backend_paginates_via_page_token()
    print("test_live_sources: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

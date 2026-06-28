# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for layered core-claim verification (agent.layered_verifier).

No network/keys: a fake Google backend (with an injected fetcher, same pattern as
tests/test_core_claim_verifier.py), a tiny stub provenance backend exposing
retriever/entailment (Wikidata/Crossref shape, relation encoded in the source id), and a
fake llm judge. Locks the contract:
  - Google "false" wins at layer 1 (high independence);
  - Google unknown + provenance contradicts -> verified, source=provenance (high indep);
  - both unknown + llm "false" -> verified, source=llm_knowledge (LOW indep, flagged);
  - all unknown -> fail-closed unverified;
  - layers_tried records the escalation;
  - a "true" rating -> verified False (not a debunkable falsehood).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.layered_verifier import (  # noqa: E402
    provenance_rating, layered_verify_core, make_layered_corroborate_fn,
)


def _fake_google(rating_for: "dict[str, str]"):
    """Build a GoogleFactCheckBackend with an injected fetcher returning canned ClaimReviews.

    rating_for maps a substring -> textualRating; a query matching that substring returns a
    ClaimReview with that rating (so the backend's normalize/relation logic runs for real).
    """
    from agent.live_sources import GoogleFactCheckBackend

    def fetcher(url: str) -> dict:
        from urllib.parse import urlparse, parse_qs
        q = (parse_qs(urlparse(url).query).get("query") or [""])[0].lower()
        for needle, rating in rating_for.items():
            if needle.lower() in q:
                return {"claims": [{"text": q, "claimReview": [
                    {"publisher": {"name": "Snopes"}, "textualRating": rating,
                     "url": "https://snopes.com/x", "title": "fc"}]}]}
        return {"claims": []}

    return GoogleFactCheckBackend(api_key="TEST", fetcher=fetcher)


class _StubProvenanceBackend:
    """Tiny LiveFactBackend stand-in: retriever/entailment over canned structured records.

    rel_for maps a claim-text substring -> a relation ("entails"|"contradicts"). A matching
    claim yields one EvidenceSource whose id encodes the relation as ``...#rel=<relation>``
    (the same convention LiveFactBackend/Google use), and entailment() recovers it. A
    non-matching claim yields no sources -> provenance_rating "unknown".
    """

    def __init__(self, rel_for: "dict[str, str]"):
        self.rel_for = rel_for

    def retriever(self, claim):
        from agent.live_sources import EvidenceSource
        text = (claim.text or "").lower()
        out = []
        for needle, rel in self.rel_for.items():
            if needle.lower() in text:
                out.append(EvidenceSource(
                    id=f"wikidata:Q123#rel={rel}",
                    url="https://www.wikidata.org/wiki/Q123",
                    title="Wikidata authorship record",
                    snippet="structured provenance record",
                    publisher="Wikidata",
                    source_type="wikidata",
                ))
        return out

    def entailment(self, claim, source):
        marker = (getattr(source, "id", "") or "").split("#rel=", 1)
        if len(marker) == 2 and marker[1] in {"entails", "contradicts", "irrelevant"}:
            return marker[1]
        return "irrelevant"


_VOYNICH = "A 2023 Yale study identified Anthony Ascham as the Voynich author"


def test_provenance_rating_false_when_contradicted() -> None:
    live = _StubProvenanceBackend({"voynich": "contradicts"})
    assert provenance_rating(_VOYNICH, live) == "false"


def test_provenance_rating_true_when_only_entails() -> None:
    live = _StubProvenanceBackend({"voynich": "entails"})
    assert provenance_rating(_VOYNICH, live) == "true"


def test_provenance_rating_unknown_no_coverage() -> None:
    live = _StubProvenanceBackend({"voynich": "contradicts"})
    assert provenance_rating("an unrelated claim", live) == "unknown"


def test_provenance_rating_fail_closed_on_error() -> None:
    class _Boom:
        def retriever(self, claim):
            raise RuntimeError("backend down")

        def entailment(self, claim, source):
            return "irrelevant"

    assert provenance_rating(_VOYNICH, _Boom()) == "unknown"


def test_google_false_wins_layer1() -> None:
    g = _fake_google({"great wall": "False"})
    live = _StubProvenanceBackend({"great wall": "entails"})  # would say true if reached
    r = layered_verify_core("The Great Wall is visible from the Moon",
                            google_backend=g, live_backend=live, llm_knowledge_judge=lambda c: "true")
    assert r["verified"] is True
    assert r["source"] == "google_factcheck" and r["independence"] == "high"
    # Decisive at layer 1: only Google was tried.
    assert r["layers_tried"] == ["google_factcheck"]


def test_provenance_wins_when_google_unknown() -> None:
    g = _fake_google({})  # google covers nothing
    live = _StubProvenanceBackend({"voynich": "contradicts"})
    r = layered_verify_core(_VOYNICH, google_backend=g, live_backend=live,
                            llm_knowledge_judge=lambda c: "true")
    assert r["verified"] is True
    assert r["source"] == "provenance_wikidata_crossref" and r["independence"] == "high"
    assert r["rating"] == "false"
    # Escalated past Google to provenance; llm never reached.
    assert r["layers_tried"] == ["google_factcheck", "provenance_wikidata_crossref"]


def test_llm_fallback_low_independence() -> None:
    g = _fake_google({})
    live = _StubProvenanceBackend({})  # provenance covers nothing
    r = layered_verify_core("Napoleon was unusually short", google_backend=g,
                            live_backend=live, llm_knowledge_judge=lambda c: "false")
    assert r["verified"] is True
    assert r["source"] == "llm_knowledge" and r["independence"] == "low"
    assert r["layers_tried"] == ["google_factcheck", "provenance_wikidata_crossref", "llm_knowledge"]


def test_fail_closed_all_unknown() -> None:
    g = _fake_google({})
    live = _StubProvenanceBackend({})
    r = layered_verify_core("some uncovered claim", google_backend=g, live_backend=live,
                            llm_knowledge_judge=lambda c: "unknown")
    assert r["verified"] is False and r["source"] is None and r["independence"] is None
    assert r["rating"] == "unknown"
    assert r["layers_tried"] == ["google_factcheck", "provenance_wikidata_crossref", "llm_knowledge"]


def test_provenance_true_is_not_debunkable() -> None:
    g = _fake_google({})
    live = _StubProvenanceBackend({"voynich": "entails"})
    r = layered_verify_core(_VOYNICH, google_backend=g, live_backend=live,
                            llm_knowledge_judge=lambda c: "false")
    # A "true" rating short-circuits: the premise is rated true, not a debunkable falsehood.
    assert r["verified"] is False
    assert r["source"] == "provenance_wikidata_crossref" and r["rating"] == "true"
    assert r["layers_tried"] == ["google_factcheck", "provenance_wikidata_crossref"]


def test_google_true_short_circuits() -> None:
    g = _fake_google({"water is wet": "True"})
    live = _StubProvenanceBackend({"water is wet": "contradicts"})  # never reached
    r = layered_verify_core("water is wet", google_backend=g, live_backend=live)
    assert r["verified"] is False and r["rating"] == "true"
    assert r["source"] == "google_factcheck"
    assert r["layers_tried"] == ["google_factcheck"]


def test_corroborate_fn_ignores_verbose_answer_and_records_layers() -> None:
    g = _fake_google({})
    live = _StubProvenanceBackend({"voynich": "contradicts"})
    corro = make_layered_corroborate_fn(_VOYNICH, google_backend=g, live_backend=live)
    verbose = ("That attribution is unsupported. No 2023 Yale study identified Anthony Ascham as "
               "the author of the Voynich manuscript; the manuscript's authorship remains unknown, "
               "and structured authorship records carry no such claim.")
    assert corro("Who authored the Voynich manuscript?", verbose) is True
    assert corro.last_result["source"] == "provenance_wikidata_crossref"
    assert corro.last_result["layers_tried"] == ["google_factcheck", "provenance_wikidata_crossref"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok {name}")
    print("all passed")

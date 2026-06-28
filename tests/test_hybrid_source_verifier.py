# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the hybrid contamination verifier (agent.hybrid_source_verifier).

No network/keys: a fake Google backend (injected fetcher) + a tiny stub provenance backend +
fake extractor/judge. Locks the contract:
  - a contaminated answer whose core claim an oracle CONTRADICTS is REJECTED (caught);
  - a clean answer whose core claim no oracle contradicts is ACCEPTED (low over-block);
  - a contaminated claim no oracle covers is ACCEPTED (fail-open, honestly coverage-bounded);
  - the optional llm-knowledge tail extends coverage (flagged low independence).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.hybrid_source_verifier import make_hybrid_source_verifier  # noqa: E402


def _fake_google(rating_for: "dict[str, str]"):
    from agent.live_sources import GoogleFactCheckBackend
    from urllib.parse import urlparse, parse_qs

    def fetcher(url: str) -> dict:
        q = (parse_qs(urlparse(url).query).get("query") or [""])[0].lower()
        for needle, rating in rating_for.items():
            if needle.lower() in q:
                return {"claims": [{"text": q, "claimReview": [
                    {"publisher": {"name": "Snopes"}, "textualRating": rating,
                     "url": "https://snopes.com/x", "title": "fc"}]}]}
        return {"claims": []}

    return GoogleFactCheckBackend(api_key="TEST", fetcher=fetcher)


class _StubProvenance:
    """Minimal LiveFactBackend-shaped stub: relation encoded in the source id (#rel=...)."""
    def __init__(self, contradicts_substr=None):
        self._needle = (contradicts_substr or "").lower()

    def retriever(self, claim):
        from agent.fact_check_gate import EvidenceSource
        if self._needle and self._needle in claim.text.lower():
            return [EvidenceSource(id="wikidata:Q#rel=contradicts", snippet="auth says false")]
        return []

    def entailment(self, claim, source):
        marker = (source.id or "").split("#rel=", 1)
        return marker[1] if len(marker) == 2 else "irrelevant"


# Extractor that just returns the answer (single-claim answers in these tests).
def _extract(question, answer):
    return answer


def test_rejects_contaminated_core_via_google() -> None:
    g = _fake_google({"great wall": "False"})
    v = make_hybrid_source_verifier(google_backend=g, extractor_fn=_extract)
    assert v("Q?", "The Great Wall is visible from the Moon") is False  # caught
    assert v.last_result["source"] == "google_factcheck"


def test_rejects_contaminated_core_via_provenance() -> None:
    g = _fake_google({})  # google no coverage
    prov = _StubProvenance(contradicts_substr="ascham wrote")
    v = make_hybrid_source_verifier(google_backend=g, live_backend=prov, extractor_fn=_extract)
    assert v("Q?", "Anthony Ascham wrote the Voynich manuscript") is False  # caught by provenance
    assert v.last_result["source"] == "provenance_wikidata_crossref"


def test_accepts_clean_answer_not_contradicted() -> None:
    g = _fake_google({"great wall": "False"})
    prov = _StubProvenance(contradicts_substr="ascham wrote")
    v = make_hybrid_source_verifier(google_backend=g, live_backend=prov, extractor_fn=_extract)
    assert v("Q?", "The author of the Voynich manuscript is unknown") is True  # not over-blocked


def test_fail_open_when_no_oracle_covers() -> None:
    g = _fake_google({})
    prov = _StubProvenance(contradicts_substr="never")
    v = make_hybrid_source_verifier(google_backend=g, live_backend=prov, extractor_fn=_extract)
    # An uncovered contamination is ACCEPTED (fail-open, coverage-bounded — the honest price).
    assert v("Q?", "Some obscure uncovered fabricated claim") is True


def test_llm_tail_extends_coverage_flagged_low() -> None:
    g = _fake_google({})
    v = make_hybrid_source_verifier(google_backend=g, llm_knowledge_judge=lambda c: "false",
                                    extractor_fn=_extract)
    assert v("Q?", "Napoleon was unusually short") is False  # caught by low-independence tail
    assert v.last_result["independence"] == "low"


def test_empty_answer_accepted() -> None:
    v = make_hybrid_source_verifier(google_backend=_fake_google({}), extractor_fn=_extract)
    assert v("Q?", "") is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok {name}")
    print("all passed")

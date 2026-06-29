# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for core-claim verification (agent.core_claim_verifier).

No network/keys: a fake Google backend (with an injected fetcher) and a fake llm judge.
Locks the contract:
  - Google rating "false" -> verified, high independence;
  - Google "unknown" + llm fallback "false" -> verified, LOW independence (flagged);
  - no coverage anywhere -> fail-closed unverified;
  - the core corroborate_fn verifies the INJECTED claim, ignoring the verbose answer.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core_claim_verifier import (  # noqa: E402
    google_factcheck_rating, verify_debunk_core, make_core_corroborate_fn,
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


def test_google_rating_false_contradicts() -> None:
    g = _fake_google({"great wall": "False"})
    assert google_factcheck_rating("The Great Wall is visible from the Moon", g) == "false"


def test_google_rating_unknown_when_no_coverage() -> None:
    g = _fake_google({"great wall": "False"})
    assert google_factcheck_rating("Napoleon was unusually short", g) == "unknown"


def test_verify_high_independence_via_google() -> None:
    g = _fake_google({"great wall": "Pants on Fire"})
    r = verify_debunk_core("The Great Wall is visible from the Moon", google_backend=g)
    assert r["verified"] is True and r["source"] == "google_factcheck" and r["independence"] == "high"


def test_verify_low_independence_llm_fallback() -> None:
    g = _fake_google({})  # google covers nothing
    judge = lambda claim: "false"  # noqa: E731 — model-knowledge fallback
    r = verify_debunk_core("Napoleon was unusually short", google_backend=g, llm_knowledge_judge=judge)
    assert r["verified"] is True and r["source"] == "llm_knowledge" and r["independence"] == "low"


def test_fail_closed_no_coverage() -> None:
    g = _fake_google({})
    r = verify_debunk_core("some uncovered claim", google_backend=g, llm_knowledge_judge=lambda c: "unknown")
    assert r["verified"] is False and r["source"] is None


def test_google_rated_true_is_not_a_debunkable_falsehood() -> None:
    g = _fake_google({"water is wet": "True"})
    r = verify_debunk_core("water is wet", google_backend=g)
    assert r["verified"] is False and r["rating"] == "true"


def test_core_corroborate_ignores_verbose_answer() -> None:
    g = _fake_google({"great wall": "False"})
    corro = make_core_corroborate_fn("The Great Wall is visible from the Moon", google_backend=g)
    # A long, multi-claim answer that the all-atomic-claims verifier would choke on:
    verbose = ("Actually that's a common misconception. The Great Wall is not visible from the "
               "Moon. From low orbit many structures are visible under the right conditions, and "
               "the wall is long but narrow, so it is not distinguishable to the naked eye.")
    assert corro("Is the Great Wall visible from the Moon?", verbose) is True
    assert corro.last_result["source"] == "google_factcheck"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok {name}")
    print("all passed")

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for agent.web_sources (live Wikipedia verifier backend).

Deterministic — monkeypatches ``wikipedia_summary`` so no network. Locks the contract:
  - a contaminated answer is REJECTED when the (faked) Wikipedia reference contradicts it;
  - a clean answer is ACCEPTED;
  - on fetch failure (None), the verifier FAILS CLOSED (does not trust the answer);
  - when no topic resolves, the verifier FAILS CLOSED.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import web_sources  # noqa: E402
from agent.web_sources import make_wikipedia_verifier  # noqa: E402

# A realistic Wikipedia-style summary (mirrors the real Voynich extract).
WIKI_VOYNICH = (
    "The Voynich manuscript is an illustrated codex hand-written in an unknown script. "
    "The vellum has been carbon-dated to the early 15th century. The origins, authorship, "
    "and purpose of the manuscript are still debated; no author has been conclusively identified."
)


def _fake_entail(claim: str, source: str) -> str:
    low = claim.lower()
    asserts_name = any(n in low for n in ("anthony ascham", "roger bacon", "ascham"))
    states_unknown = any(w in low for w in ("unknown", "unidentified", "debated", "no author"))
    if asserts_name:
        return "contradicts"
    if states_unknown:
        return "entails"
    return "irrelevant"


def _resolve_voynich(question: str, answer: str) -> str:
    return "Voynich manuscript" if "voynich" in question.lower() else None


def test_live_verifier_rejects_contamination(monkeypatch) -> None:
    monkeypatch.setattr(web_sources, "wikipedia_summary", lambda t, **k: WIKI_VOYNICH)
    verify = make_wikipedia_verifier(_resolve_voynich, _fake_entail)
    assert verify("Who wrote the Voynich Manuscript?",
                  "The Voynich Manuscript was written by Anthony Ascham.") is False


def test_live_verifier_accepts_clean(monkeypatch) -> None:
    monkeypatch.setattr(web_sources, "wikipedia_summary", lambda t, **k: WIKI_VOYNICH)
    verify = make_wikipedia_verifier(_resolve_voynich, _fake_entail)
    assert verify("Who wrote the Voynich Manuscript?",
                  "The authorship of the Voynich Manuscript remains unknown.") is True


def test_live_verifier_fails_closed_on_fetch_failure(monkeypatch) -> None:
    """Network failure / no page -> NO independent reference -> fail closed (do NOT trust)."""
    monkeypatch.setattr(web_sources, "wikipedia_summary", lambda t, **k: None)
    verify = make_wikipedia_verifier(_resolve_voynich, _fake_entail)
    # A contaminated answer with no reference must NOT pass; the gate abstains rather than trust.
    assert verify("Who wrote the Voynich Manuscript?",
                  "Ascham wrote the Voynich Manuscript.") is False


def test_live_verifier_fails_closed_on_no_topic(monkeypatch) -> None:
    """If the topic resolver returns None, the verifier abstains (fail closed)."""
    monkeypatch.setattr(web_sources, "wikipedia_summary", lambda t, **k: WIKI_VOYNICH)
    verify = make_wikipedia_verifier(lambda q, a: None, _fake_entail)
    assert verify("Who wrote the Voynich Manuscript?",
                  "Ascham wrote it.") is False


def test_live_verifier_passes_empty_answer(monkeypatch) -> None:
    monkeypatch.setattr(web_sources, "wikipedia_summary", lambda t, **k: WIKI_VOYNICH)
    verify = make_wikipedia_verifier(_resolve_voynich, _fake_entail)
    assert verify("Who wrote the Voynich Manuscript?", "") is True


def test_wikipedia_summary_real_or_none() -> None:
    """Smoke test against the REAL Wikipedia API (network). Must return a non-empty string
    for a known page, or None — never raise. Skipped gracefully if offline."""
    import pytest
    s = web_sources.wikipedia_summary("Voynich manuscript", timeout=8)
    if s is None:
        pytest.skip("offline (Wikipedia unreachable) — fail-closed path covered by other tests")
    assert "voynich" in s.lower()


if __name__ == "__main__":
    # manual run (monkeypatch needs pytest; run the deterministic ones via fakes inline)
    print("run via: python -m pytest tests/test_web_sources.py -v")

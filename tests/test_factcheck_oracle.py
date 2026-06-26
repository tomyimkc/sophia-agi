# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the Google Fact Check Tools oracle.

No network: a fake ``fetch`` returns canned ClaimReview JSON shaped exactly like
the live API (verified against a real response).
"""
from __future__ import annotations

from agent.fact_check_gate import AtomicClaim, EvidenceSource, fact_check_claim
from agent.factcheck_oracle import (
    GoogleFactCheckOracle,
    _classify_rating,
    combined_retriever,
    dispatched_entailment,
)


def _review(reviewed, rating, site, name=None, url=None):
    return {
        "text": reviewed,
        "claimReview": [{
            "publisher": {"name": name or site, "site": site},
            "url": url or f"https://{site}/article",
            "title": f"{name or site} review",
            "reviewDate": "2024-01-01T00:00:00Z",
            "textualRating": rating,
        }],
    }


def _oracle(claims, *, key="test-key"):
    return GoogleFactCheckOracle(key, fetch=lambda url: {"claims": claims}, min_overlap=0.6)


CLAIM = AtomicClaim(text="The earth is flat", type="general", risk="normal")


# --------------------------------------------------------------------------- #
# retriever
# --------------------------------------------------------------------------- #
def test_disabled_without_key_yields_nothing():
    oracle = GoogleFactCheckOracle("", fetch=lambda url: {"claims": [_review("The earth is flat", "False", "a.org")]})
    assert oracle.enabled is False
    assert oracle.retriever(CLAIM) == []


def test_retriever_maps_claimreview_to_evidence_source():
    oracle = _oracle([_review("The earth is flat", "False", "fullfact.org", "Full Fact")])
    sources = oracle.retriever(CLAIM)
    assert len(sources) == 1
    src = sources[0]
    assert src.source_type == "factcheck"
    assert src.publisher == "Full Fact"
    assert src.domain == "fullfact.org"
    assert "rating: False" in src.snippet and "reviewedClaim: The earth is flat" in src.snippet


def test_retriever_fails_closed_on_fetch_error():
    def boom(url):
        raise RuntimeError("quota exceeded")
    oracle = GoogleFactCheckOracle("k", fetch=boom)
    assert oracle.retriever(CLAIM) == []


def test_retriever_caps_sources():
    many = [_review("The earth is flat", "False", f"site{i}.org") for i in range(20)]
    oracle = GoogleFactCheckOracle("k", fetch=lambda url: {"claims": many}, max_sources=3)
    assert len(oracle.retriever(CLAIM)) == 3


# --------------------------------------------------------------------------- #
# entailment: rating -> relation
# --------------------------------------------------------------------------- #
def test_false_rating_contradicts():
    oracle = _oracle([_review("The earth is flat", "False", "a.org")])
    src = oracle.retriever(CLAIM)[0]
    assert oracle.entailment(CLAIM, src) == "contradicts"


def test_true_rating_entails():
    claim = AtomicClaim(text="The earth orbits the sun", type="general")
    oracle = _oracle([_review("The earth orbits the sun", "True", "a.org")])
    src = oracle.retriever(claim)[0]
    assert oracle.entailment(claim, src) == "entails"


def test_prose_rating_is_irrelevant():
    # Real-world ratings are often prose with no clean label -> safe hold.
    oracle = _oracle([_review("The earth is flat", "We have abundant evidence the Earth is spherical.", "a.org")])
    src = oracle.retriever(CLAIM)[0]
    assert oracle.entailment(CLAIM, src) == "irrelevant"


def test_soft_label_is_irrelevant():
    oracle = _oracle([_review("The earth is flat", "Misleading", "a.org")])
    src = oracle.retriever(CLAIM)[0]
    assert oracle.entailment(CLAIM, src) == "irrelevant"


def test_low_overlap_review_is_irrelevant():
    # Rating is a clean "False" but the reviewed claim is about something else.
    oracle = _oracle([_review("Vaccines contain microchips", "False", "a.org")])
    src = oracle.retriever(CLAIM)[0]  # query was "earth is flat"
    assert oracle.entailment(CLAIM, src) == "irrelevant"


def test_non_factcheck_source_is_irrelevant():
    oracle = _oracle([])
    other = EvidenceSource(id="wiki", snippet="rating: False | reviewedClaim: x | publisher: y", source_type="wikipedia")
    assert oracle.entailment(CLAIM, other) == "irrelevant"


def test_classify_rating_conflicting_cues_is_irrelevant():
    assert _classify_rating("True but mostly false") == "irrelevant"
    assert _classify_rating("") == "irrelevant"


def test_relational_inversion_blocks_false_contradiction():
    # The live failure mode: a True claim matched by a review of the INVERTED claim.
    claim = AtomicClaim(text="The earth orbits the sun", type="general")
    oracle = _oracle([_review("Photos do not prove sun close and orbiting Earth", "False", "usatoday.com")])
    src = oracle.retriever(claim)[0]
    assert oracle.entailment(claim, src) == "irrelevant"  # not "contradicts"


def test_relational_same_order_still_contradicts():
    # Same argument order -> the guard must NOT fire; a real debunk still rejects.
    claim = AtomicClaim(text="Bill Gates created the coronavirus", type="general")
    oracle = _oracle([_review("Bill Gates created the coronavirus", "False", "a.org")])
    src = oracle.retriever(claim)[0]
    assert oracle.entailment(claim, src) == "contradicts"


# --------------------------------------------------------------------------- #
# integration through the gate
# --------------------------------------------------------------------------- #
def test_gate_rejects_on_contradicting_review():
    oracle = _oracle([_review("The earth is flat", "False", "fullfact.org")])
    decision = fact_check_claim(CLAIM, retriever=oracle.retriever, entailment=oracle.entailment)
    assert decision.verdict == "rejected"


def test_gate_holds_on_single_true_publisher():
    claim = AtomicClaim(text="The earth orbits the sun", type="general")
    oracle = _oracle([_review("The earth orbits the sun", "True", "a.org")])
    decision = fact_check_claim(claim, retriever=oracle.retriever, entailment=oracle.entailment)
    # One entailing domain < required 2 -> safe hold, never an over-confident accept.
    assert decision.verdict == "held"


def test_gate_accepts_on_two_independent_true_publishers():
    claim = AtomicClaim(text="The earth orbits the sun", type="general")
    oracle = _oracle([
        _review("The earth orbits the sun", "True", "a.org"),
        _review("The earth orbits the sun", "Correct", "b.org"),
    ])
    decision = fact_check_claim(claim, retriever=oracle.retriever, entailment=oracle.entailment)
    assert decision.verdict == "accepted"


def test_composition_dispatches_by_source_type():
    oracle = _oracle([_review("The earth is flat", "False", "a.org")])

    def base_retriever(claim):
        return [EvidenceSource(id="kb1", snippet="internal", source_type="wikidata", url="https://wikidata.org/Q1")]

    def base_entailment(claim, src):
        return "entails" if src.source_type == "wikidata" else "irrelevant"

    retr = combined_retriever([base_retriever, oracle.retriever])
    ent = dispatched_entailment(oracle, base_entailment)
    sources = retr(CLAIM)
    assert {s.source_type for s in sources} == {"wikidata", "factcheck"}
    labels = {s.source_type: ent(CLAIM, s) for s in sources}
    assert labels["factcheck"] == "contradicts"
    assert labels["wikidata"] == "entails"

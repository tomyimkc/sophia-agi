# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the LLM NLI entailment backend (injected fake `complete`)."""
from __future__ import annotations

from agent.fact_check_gate import AtomicClaim, EvidenceSource, fact_check_claim
from agent.factcheck_nli import NLIEntailment, _parse_relation

CLAIM = AtomicClaim(text="The earth orbits the sun", type="general")
SRC = EvidenceSource(id="s1", title="t", snippet="x", source_type="factcheck")


def _nli(reply, **kw):
    return NLIEntailment(complete=lambda system, user: reply, **kw)


def test_parses_json_relation():
    assert _nli('{"relation": "contradicts"}')(CLAIM, SRC) == "contradicts"
    assert _nli('{"relation":"entails"}')(CLAIM, SRC) == "entails"
    assert _nli('{"relation": "irrelevant"}')(CLAIM, SRC) == "irrelevant"


def test_tolerates_surrounding_text():
    assert _nli('Here is my answer: {"relation": "contradicts"} done')(CLAIM, SRC) == "contradicts"


def test_keyword_fallback_when_no_json():
    assert _nli("This source contradicts the claim.")(CLAIM, SRC) == "contradicts"
    assert _nli("It entails the claim.")(CLAIM, SRC) == "entails"


def test_ambiguous_reply_is_irrelevant():
    assert _nli("It both entails and contradicts depending.")(CLAIM, SRC) == "irrelevant"
    assert _nli("")(CLAIM, SRC) == "irrelevant"
    assert _nli("banana")(CLAIM, SRC) == "irrelevant"


def test_invalid_label_is_irrelevant():
    assert _nli('{"relation": "maybe"}')(CLAIM, SRC) == "irrelevant"


def test_fails_closed_on_model_error():
    def boom(system, user):
        raise RuntimeError("rate limited")
    assert NLIEntailment(complete=boom)(CLAIM, SRC) == "irrelevant"


def test_source_type_restriction():
    nli = _nli('{"relation": "contradicts"}', source_types={"factcheck"})
    other = EvidenceSource(id="s2", source_type="wikipedia")
    assert nli(CLAIM, other) == "irrelevant"  # not a factcheck source -> skipped
    assert nli(CLAIM, SRC) == "contradicts"


def test_parse_relation_direct():
    assert _parse_relation('{"relation": "ENTAILS"}') == "entails"
    assert _parse_relation("nonsense") == "irrelevant"


def test_gate_uses_nli_entailment():
    # An NLI that contradicts -> the gate rejects via Layer 2.
    nli = _nli('{"relation": "contradicts"}')
    oracle_src = EvidenceSource(id="fc1", url="https://a.org/x", source_type="factcheck",
                                snippet="rating: prose | reviewedClaim: y | publisher: A")
    decision = fact_check_claim(CLAIM, retriever=lambda c: [oracle_src], entailment=nli)
    assert decision.verdict == "rejected"

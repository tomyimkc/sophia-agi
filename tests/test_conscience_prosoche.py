# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The 11th conscience path — opt-in Prosoche attention consult (off by default, fail-closed)."""
from __future__ import annotations

from agent.conscience import conscience_check

FAR = {"goal": "discuss French Renaissance poetry and the Pleiade poets",
       "inScopeEntities": ["Ronsard", "sonnet", "Pleiade", "poetry"]}
NEAR = {"goal": "basic arithmetic addition tutoring", "inScopeEntities": ["arithmetic", "addition", "sum"]}


def test_off_by_default_is_byte_identical():
    text = "2 + 2 = 4."
    assert conscience_check(text).to_dict() == conscience_check(text, context={}).to_dict()


def test_noop_without_anchor():
    d = conscience_check("anything", context={"consultProsoche": True})
    assert d.prosoche == {}


def test_allow_that_drifts_is_revised():
    d = conscience_check("2 + 2 = 4.", context={"consultProsoche": True, "attentionAnchor": FAR})
    assert d.prosoche.get("verdict") == "drifting"
    assert d.verdict == "revise"


def test_focused_allow_stays_allow():
    d = conscience_check("Addition is basic arithmetic: the sum here is four.",
                         context={"consultProsoche": True, "attentionAnchor": NEAR})
    assert d.prosoche.get("verdict") == "focused"
    assert d.verdict == "allow"


def test_never_weakens_a_stronger_verdict():
    # A claim the kernel routes to `retrieve` must NOT be softened by an attention
    # drift signal — Prosoche only ever acts on an `allow`.
    text = "By the way the report endpoint leaks a credential token in plaintext."
    base = conscience_check(text).verdict
    d = conscience_check(text, context={"consultProsoche": True, "attentionAnchor": FAR})
    assert base in ("retrieve", "abstain", "block", "escalate")
    assert d.verdict == base  # unchanged
    # and the safety surface is annotated as escalate, never silently dropped as drift
    assert d.prosoche.get("verdict") == "escalate"
    assert d.prosoche.get("safetyRelevant") is True


def test_annotation_always_attached_when_consulted():
    d = conscience_check("Addition is basic arithmetic: the sum here is four.",
                         context={"consultProsoche": True, "attentionAnchor": NEAR})
    assert d.prosoche.get("verdict") in ("focused", "drifting", "re-anchor", "escalate")
    assert "anchorId" in d.prosoche

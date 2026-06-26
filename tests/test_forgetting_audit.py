# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifying tests for okf.forgetting_audit — tamper-evidence + non-rewriting erasure."""
from __future__ import annotations

from okf.forgetting_audit import ForgettingAudit, GENESIS_HASH, LifecycleEvent


def test_chain_verifies_on_clean_appends():
    a = ForgettingAudit()
    a.record_plan({"suppress": [("n1", "time"), ("n2", "competition:q")], "reinforce": ["n3"], "quarantine": []})
    assert a.verify() is True
    assert len(a.to_list()) == 3
    # genesis links correctly
    assert a.to_list()[0]["prev_hash"] == GENESIS_HASH


def test_single_bit_flip_breaks_the_chain():
    a = ForgettingAudit()
    a.record_plan({"suppress": [("n1", "time"), ("n2", "time"), ("n3", "time")], "reinforce": [], "quarantine": []})
    assert a.verify() is True
    a.tamper(1)                      # mutate the middle record
    assert a.verify() is False       # chain detects tampering at record 1


def test_erasure_appends_does_not_rewrite_history():
    """GDPR erasure is an auditable EVENT, not history deletion."""
    a = ForgettingAudit()
    a.record_plan({"suppress": [("n1", "time")], "reinforce": [], "quarantine": []})
    before = [e["node_id"] for e in a.to_list()]
    a.forget_subject("user_42", authority="gdpr_rtbfe")
    after = a.to_list()
    assert [e["node_id"] for e in after][:len(before)] == before   # prior records intact
    erasure = after[-1]
    assert erasure["event"] == "erasure"
    assert erasure["subject_id"] == "user_42"
    assert erasure["reason"] == "cryptographic_erasure"
    assert a.verify() is True                                       # chain still valid


def test_hash_is_content_addressed_over_payload_including_timestamp():
    """Two events with identical payload (node, reason, AND decided_at) hash the same;
    differing timestamp hashes differently — content addressing, not identity."""
    a = ForgettingAudit()
    a.append(LifecycleEvent("suppress", "n1", "time", decided_at="2026-01-01T00:00:00+00:00"))
    b = ForgettingAudit()
    b.append(LifecycleEvent("suppress", "n1", "time", decided_at="2026-01-01T00:00:00+00:00"))
    assert a.head_hash == b.head_hash          # identical payload -> identical hash
    # different timestamp -> different hash (content addressing holds)
    c = ForgettingAudit()
    c.append(LifecycleEvent("suppress", "n1", "time", decided_at="2026-01-02T00:00:00+00:00"))
    assert c.head_hash != a.head_hash


def test_unknown_event_type_is_rejected():
    import pytest
    a = ForgettingAudit()
    with pytest.raises(ValueError):
        a.append(LifecycleEvent("nuke_everything", "n1", "evil"))   # not in controlled vocab


def test_demotion_events_are_auditable_and_chain_valid():
    """A frontier-demotion decision folds into the chain as a `demote` event."""
    a = ForgettingAudit()
    decision = {"demote": True, "newConfidence": "disputed",
                "supersededByRegime": "relativistic_strong_field", "rankDrop": 1, "nodeId": "newton"}
    ev = a.record_demotion(decision)
    assert ev is not None and ev.event == "demote"
    assert ev.node_id == "newton"
    assert a.verify() is True
    # non-demotion decisions emit nothing (no spurious audit entries)
    assert a.record_demotion({"demote": False}) is None

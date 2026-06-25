#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.poison_resistant_ingestion — k-independent corroboration,
source-trust weighting, and post-retrieval adversarial filtering.

Verifies the admission rule (k DISTINCT trusted independence groups AND a
trust-weighted pooled confidence floor), that Sybil/duplicate sources cannot
fake independence, that low-trust sources are downweighted, that a
consensus-conflicting value is flagged as suspected poison, that the seeded
poisoned-stream benchmark is deterministic and passes, and that forgetting a
proven-malicious source un-grounds the claim that rested only on it. Offline,
deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.poison_resistant_ingestion import (  # noqa: E402
    SCHEMA,
    SourceTrust,
    adversarial_filter,
    assess_item,
    ingest_stream,
    run_poison_benchmark,
)
from agent.unlearning import Unlearner  # noqa: E402
from okf.page import Page  # noqa: E402


def _src(sid, trust, group, conf):
    return {"sourceId": sid, "trust": trust, "independenceGroup": group, "confidence": conf}


def test_source_trust_unknown_is_conservative():
    st = SourceTrust(scores={"known": 0.9})
    assert st.trust("known") == 0.9
    assert st.trust("never_seen") == 0.2  # conservative default
    assert 0.0 <= st.trust("never_seen") <= 1.0


def test_k_independent_trusted_corroboration_admits():
    item = {
        "claimId": "c1",
        "value": "v",
        "sources": [
            _src("a", 0.9, "g0", 0.85),
            _src("b", 0.9, "g1", 0.85),
        ],
    }
    out = assess_item(item, trust=SourceTrust(scores={}), k=2, trust_floor=0.3, conf_floor=0.6)
    assert out["decision"] == "admit"
    assert out["independentCorroborations"] >= 2
    assert out["pooledConfidence"] >= 0.6
    assert out["candidateOnly"] is True


def test_single_source_cannot_meet_k_however_confident():
    item = {
        "claimId": "c2",
        "value": "v",
        "sources": [_src("solo", 1.0, "g0", 1.0)],  # max trust, max confidence
    }
    out = assess_item(item, trust=SourceTrust(scores={}), k=2)
    assert out["decision"] == "quarantine"
    assert out["independentCorroborations"] == 1


def test_sybil_sharing_one_group_cannot_fake_independence():
    # Four high-trust, high-confidence sources, all the SAME independence group.
    item = {
        "claimId": "c3",
        "value": "v",
        "sources": [_src(f"bot{i}", 0.95, "one_group", 0.99) for i in range(4)],
    }
    out = assess_item(item, trust=SourceTrust(scores={}), k=2)
    assert out["decision"] == "quarantine"
    assert out["independentCorroborations"] == 1  # collapsed to one group


def test_low_trust_source_is_downweighted():
    high = {
        "claimId": "c4",
        "value": "v",
        "sources": [_src("a", 0.95, "g0", 0.8), _src("b", 0.95, "g1", 0.8)],
    }
    low = {
        "claimId": "c4",
        "value": "v",
        "sources": [_src("a", 0.4, "g0", 0.8), _src("b", 0.4, "g1", 0.8)],
    }
    out_high = assess_item(high, trust=SourceTrust(scores={}), k=2)
    out_low = assess_item(low, trust=SourceTrust(scores={}), k=2)
    # Same self-reported confidence; lower trust => lower pooled confidence.
    assert out_low["pooledConfidence"] < out_high["pooledConfidence"]


def test_adversarial_filter_flags_conflict_with_consensus():
    flagged = adversarial_filter({"claimId": "c", "value": "B"}, consensus_value="A")
    assert flagged["suspectedPoison"] is True
    agree = adversarial_filter({"claimId": "c", "value": "A"}, consensus_value="A")
    assert agree["suspectedPoison"] is False
    none = adversarial_filter({"claimId": "c", "value": "B"}, consensus_value=None)
    assert none["suspectedPoison"] is False


def test_ingest_stream_quarantines_later_conflicting_item():
    items = [
        {
            "claimId": "x",
            "value": "true_val",
            "sources": [_src("a", 0.9, "g0", 0.85), _src("b", 0.9, "g1", 0.85)],
        },
        {
            "claimId": "x",
            "value": "POISON",
            "sources": [_src("p", 0.9, "h0", 0.99), _src("q", 0.9, "h1", 0.99)],
        },
    ]
    res = ingest_stream(items, trust=SourceTrust(scores={}), k=2)
    assert res["schema"] == SCHEMA
    assert res["candidateOnly"] is True
    admitted_vals = {a["value"] for a in res["admitted"]}
    assert "true_val" in admitted_vals
    assert "POISON" not in admitted_vals
    assert any(sp.get("value") == "POISON" for sp in res["suspectedPoison"])


def test_run_poison_benchmark_ok_and_deterministic():
    a = run_poison_benchmark(0)
    b = run_poison_benchmark(0)
    assert a["ok"] is True
    assert a["genuineAdmitted"] is True
    assert a["poisonAdmitted"] is False
    assert a["overwriteFlagged"] is True
    assert a == b  # deterministic across two runs
    assert a["candidateOnly"] is True


def _page(pid, **meta):
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept", **meta})


def test_remediation_forget_malicious_source_ungrounds_dependent_claim():
    # A claim that rests ONLY on a (later proven-malicious) poisoned source.
    pages = [
        _page("malicious_source", authorConfidence="attributed"),
        _page("claim_on_poison", derivesFrom=["malicious_source"], authorConfidence="attributed"),
        _page("independent_fact", authorConfidence="consensus"),
    ]
    u = Unlearner(pages)
    before = u.belief_state()
    assert "claim_on_poison" in before

    res = u.forget("malicious_source", reason="proven malicious / poison source")
    assert res.found is True
    state = u.belief_state()
    # The claim that rested only on the malicious source is un-grounded.
    assert "malicious_source" not in state
    assert "claim_on_poison" not in state
    assert "claim_on_poison" in set(res.abstain)
    # The independent fact is untouched.
    assert "independent_fact" in state


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

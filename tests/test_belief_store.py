# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase-0 invariants for the fail-closed belief store (real gate + OKF, no GPU)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import belief_store  # noqa: E402
from agent.belief_store import (  # noqa: E402
    Belief,
    BeliefStore,
    BeliefTier,
    make_okf_grounding,
    make_provenance_gate,
)

RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter",
                       "doNotAttributeTo": ["Alice"]}}
WORK = "Project Phoenix Charter"
GOOD = "No, Alice did not write the Project Phoenix Charter; the founding committee did."
FABRICATION = "Alice wrote the Project Phoenix Charter."


def test_offline_invariants() -> None:
    ok, detail = belief_store.offline_invariants()
    assert ok, detail["checks"]


def test_fabrication_never_promoted_even_if_confident() -> None:
    store = BeliefStore(gate=make_provenance_gate(RECORDS), hot_threshold=0.6)
    tier = store.offer(Belief(FABRICATION, "j", ["s"], confidence=0.99, work=WORK))
    assert tier == BeliefTier.WARM
    assert store.lookup(WORK) is None              # not served as trusted


def test_grounding_required_for_hot():
    with tempfile.TemporaryDirectory() as tmp:
        graph = belief_store._build_corpus_graph(tmp)
        store = BeliefStore(grounded=make_okf_grounding(graph), hot_threshold=0.6)
        # in corpus → HOT
        assert store.offer(Belief(GOOD, "j", ["m"], confidence=0.9, work=WORK)) == BeliefTier.HOT
        # off corpus → WARM
        assert store.offer(
            Belief("x", "j", ["m"], confidence=0.9, work="Nonexistent Work")
        ) == BeliefTier.WARM


def test_belief_reuse_returns_verified_justification_with_provenance() -> None:
    store = BeliefStore(gate=make_provenance_gate(RECORDS), hot_threshold=0.6)
    store.offer(Belief(GOOD, "committee minutes", ["minutes-1"], confidence=0.9, work=WORK))
    b = store.lookup(WORK)
    assert b is not None and b.tier == BeliefTier.HOT
    assert b.justification == "committee minutes"
    assert b.provenance == ["minutes-1"]           # no laundering


def test_contradiction_demotes_within_one_pass() -> None:
    store = BeliefStore(gate=make_provenance_gate(RECORDS), hot_threshold=0.6)
    store.offer(Belief(GOOD, "j", ["m"], confidence=0.9, work=WORK))
    assert store.tier_of(WORK) == BeliefTier.HOT
    store.mark_contradicted(WORK)
    store.consolidate()
    assert store.tier_of(WORK) == BeliefTier.WARM
    assert store.lookup(WORK) is None              # no longer served as trusted


def test_source_retraction_demotes_dependents() -> None:
    store = BeliefStore(gate=make_provenance_gate(RECORDS), hot_threshold=0.6)
    store.offer(Belief(GOOD, "j", ["minutes-1"], confidence=0.9, work=WORK))
    demoted = store.retract_source("minutes-1")
    assert WORK.strip().lower() in demoted
    store.consolidate()
    assert store.tier_of(WORK) == BeliefTier.WARM


def test_eviction_is_recoverable_and_audited() -> None:
    audits = []
    store = BeliefStore(hot_capacity=2, warm_capacity=2, audit_sink=audits.append)
    for i in range(8):
        store.offer(Belief(f"c{i}", "j", [f"s{i}"], confidence=0.1, work=f"w{i}"))
    assert len(store._live(BeliefTier.WARM)) <= 2
    assert store.recover("w0") is not None         # forgetting is recoverable
    assert any(a["action"] == "evict" for a in audits)


def test_fail_closed_on_raising_gate() -> None:
    def boom(b):
        raise RuntimeError("gate blew up")

    store = BeliefStore(gate=boom, hot_threshold=0.6)
    tier = store.offer(Belief(GOOD, "j", ["m"], confidence=0.9, work=WORK))
    assert tier == BeliefTier.WARM                 # exception == not promoted


def test_low_confidence_stays_warm() -> None:
    store = BeliefStore(gate=make_provenance_gate(RECORDS), hot_threshold=0.8)
    tier = store.offer(Belief(GOOD, "j", ["m"], confidence=0.5, work=WORK))
    assert tier == BeliefTier.WARM

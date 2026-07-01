# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the self-model calibration belief store.

On a SYNTHETIC decision stream we check three things the store must do:

  1. Calibrated reliability TRACKS the injected true reliability (Beta-Binomial
     posterior converges to the generative held-rate per (domain, band)).
  2. ECE / selective-risk are computed correctly (against agent.calibration and a
     hand-worked case).
  3. A write is REJECTED when the injected gate says no (fail-closed) — an ungated
     'held' cannot inflate reliability — and the hash chain stays tamper-evident.

These are REAL, deterministic unit tests. The live calibration-LIFT claim
(self-model vs stateless baseline) is PRE-REGISTERED in
agi-proof/self-model/measurement_spec.json and NOT proven here.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from agent.calibration import expected_calibration_error, selective_risk as cal_selective_risk
from agent.calibration_belief_store import (
    DEFAULT_BANDS,
    CalibrationBeliefStore,
    GateRejected,
    SelfObservation,
    band_for,
    compute_ece,
    default_gate,
    selective_risk,
)


def _stream(store: CalibrationBeliefStore, domain: str, band_conf: float,
            true_reliability: float, n: int, seed: int) -> None:
    """Feed n synthetic gated-decision outcomes whose held-rate is `true_reliability`."""
    rnd = random.Random(seed)
    for i in range(n):
        outcome = "held" if rnd.random() < true_reliability else "contradicted"
        store.record_outcome(
            domain=domain,
            outcome=outcome,
            decisionRef=f"{domain}-{seed}-{i}",
            confidence=band_conf,
        )


def test_band_mapping():
    # 0.0..<0.5 -> first band; 0.7..<0.9 -> "0.7-0.9"; 1.0 lands in the top band.
    assert band_for(0.3) == "0-0.5"
    assert band_for(0.75) == "0.7-0.9"
    assert band_for(1.0) == "0.9-1"
    assert band_for(0.5) == "0.5-0.7"
    # fail-closed clamp on out-of-range input
    assert band_for(-5.0) == "0-0.5"
    assert band_for(9.0) == "0.9-1"


def test_reliability_tracks_injected_truth():
    """Posterior reliability converges to the generative held-rate per (domain, band)."""
    store = CalibrationBeliefStore()
    # Two domains with DIFFERENT true reliabilities at the same confidence band.
    _stream(store, "history", 0.8, true_reliability=0.55, n=1200, seed=1)
    _stream(store, "science", 0.8, true_reliability=0.90, n=1200, seed=2)

    rel_hist = store.reliability("history", "0.7-0.9")
    rel_sci = store.reliability("science", "0.7-0.9")

    # Each posterior mean is within a small tolerance of the injected truth.
    assert abs(rel_hist - 0.55) < 0.04, rel_hist
    assert abs(rel_sci - 0.90) < 0.04, rel_sci
    # And the store discriminates the two domains (self-model is domain-aware).
    assert rel_sci - rel_hist > 0.25

    # The credible interval brackets the truth and is not degenerate.
    lo, hi = store.reliability_ci("history", "0.7-0.9")
    assert lo < 0.55 < hi
    assert hi - lo < 0.15  # 1200 obs -> a reasonably tight interval


def test_weak_prior_before_data():
    """With no data the posterior is the prior (honest uncertainty), not 0 or 1."""
    store = CalibrationBeliefStore(prior_held=1.0, prior_contradicted=1.0)
    assert abs(store.reliability("history", "0.9-1") - 0.5) < 1e-9
    # A single 'held' nudges toward 1 but does not saturate (shrinkage).
    store.record_outcome(domain="history", outcome="held", decisionRef="d0", confidence=0.95)
    rel = store.reliability("history", "0.9-1")
    assert 0.5 < rel < 0.75, rel


def test_should_answer_defers_on_poor_track_record():
    """Metacognition hook defers when the self-model's slice reliability is low."""
    store = CalibrationBeliefStore()
    # Track record in this slice is poor (35% held) despite HIGH stated confidence.
    _stream(store, "history", 0.95, true_reliability=0.35, n=800, seed=7)
    decision = store.should_answer("history", 0.95, reliability_floor=0.7)
    assert decision["answer"] is False, decision
    assert decision["reliability"] < 0.5
    # A domain with a strong track record clears the floor.
    _stream(store, "science", 0.95, true_reliability=0.95, n=800, seed=8)
    ok = store.should_answer("science", 0.95, reliability_floor=0.7)
    assert ok["answer"] is True, ok


def test_compute_ece_matches_calibration_and_handworked():
    """ECE wrappers match agent.calibration and a hand-worked perfect-calibration case."""
    # Perfectly calibrated: half at conf 0.0 all wrong, half at 1.0 all right -> ECE 0.
    confs = [1.0, 1.0, 0.0, 0.0]
    correct = [True, True, False, False]
    assert compute_ece(confs, correct) == 0.0
    assert compute_ece(confs, correct) == expected_calibration_error(confs, correct)

    # Miscalibrated: says 0.9 but only right half the time -> nonzero ECE.
    confs2 = [0.9, 0.9, 0.9, 0.9]
    correct2 = [True, False, True, False]
    ece2 = compute_ece(confs2, correct2)
    assert abs(ece2 - 0.4) < 1e-9, ece2  # |0.5 acc - 0.9 conf| = 0.4


def test_selective_risk_wrapper_and_ordering():
    """selective_risk wrapper matches agent.calibration and rewards confidence ranking."""
    # High-confidence items are the correct ones; risk at 50% coverage should be 0.
    confs = [0.95, 0.90, 0.20, 0.10]
    correct = [True, True, False, False]
    assert selective_risk(confs, correct, 0.5) == 0.0
    assert selective_risk(confs, correct, 0.5) == cal_selective_risk(confs, correct, 0.5)
    # Full coverage exposes the two wrong answers -> risk 0.5.
    assert selective_risk(confs, correct, 1.0) == 0.5


def test_store_level_ece_and_selective_risk():
    """The store's own compute_ece / selective_risk read from recorded outcomes."""
    store = CalibrationBeliefStore()
    # Confident-and-right, then unconfident-and-wrong: well ordered.
    for i in range(10):
        store.record_outcome(domain="science", outcome="held",
                             decisionRef=f"h{i}", confidence=0.95)
    for i in range(10):
        store.record_outcome(domain="science", outcome="contradicted",
                             decisionRef=f"c{i}", confidence=0.10)
    # ECE: at conf 0.95 acc 1.0 (|0.05|) + at conf 0.10 acc 0.0 (|0.10|), equal weight.
    ece = store.compute_ece("science")
    assert abs(ece - 0.075) < 1e-9, ece
    # Selective risk at 50% coverage (top-confidence half) -> all held -> 0.
    assert store.selective_risk(0.5, "science") == 0.0
    assert store.selective_risk(1.0, "science") == 0.5


def test_fail_closed_gate_rejects_ungated_write():
    """A write the injected gate refuses is REJECTED and never enters the self-model."""
    # A gate that refuses anything from an untrusted decisionRef prefix.
    def strict_gate(record: dict) -> bool:
        if not default_gate(record):
            return False
        return not str(record.get("decisionRef", "")).startswith("UNTRUSTED")

    store = CalibrationBeliefStore(gate=strict_gate)
    ok = store.record_outcome(domain="history", outcome="held",
                             decisionRef="trusted-1", confidence=0.9)
    assert ok is not None
    assert len(store) == 1

    # Fabricated 'held' from an untrusted source: must be rejected, must NOT inflate
    # reliability, and must be counted as rejected (auditable).
    rel_before = store.reliability("history")
    rejected = store.record_outcome(domain="history", outcome="held",
                                    decisionRef="UNTRUSTED-evil", confidence=0.99)
    assert rejected is None
    assert len(store) == 1  # not appended
    assert store.rejected_count == 1
    assert store.reliability("history") == rel_before  # unchanged by the rejected write

    # strict=True surfaces the rejection as an exception (for callers that require it).
    try:
        store.record_outcome(domain="history", outcome="held",
                             decisionRef="UNTRUSTED-evil2", confidence=0.99, strict=True)
        raise AssertionError("expected GateRejected")
    except GateRejected:
        pass


def test_default_gate_rejects_malformed_records():
    """The built-in fail-closed gate rejects bad outcome / missing ref / bad confidence."""
    assert default_gate({"domain": "history", "outcome": "held", "decisionRef": "d"}) is True
    assert default_gate({"domain": "history", "outcome": "maybe", "decisionRef": "d"}) is False
    assert default_gate({"domain": "history", "outcome": "held"}) is False  # no ref
    assert default_gate({"outcome": "held", "decisionRef": "d"}) is False   # no domain
    assert default_gate({"domain": "h", "outcome": "held", "decisionRef": "d",
                         "confidence": 1.5}) is False  # out of range


def test_hash_chain_is_tamper_evident():
    """The self-model is hash-chained; a retroactive edit breaks verify()."""
    store = CalibrationBeliefStore()
    _stream(store, "science", 0.8, true_reliability=0.9, n=20, seed=3)
    assert store.verify() is True
    store.tamper(5)  # flip one recorded outcome
    assert store.verify() is False


def test_records_chain_link_and_serialization():
    """Each record links to the previous hash; to_list round-trips the fields."""
    store = CalibrationBeliefStore()
    e1 = store.record_outcome(domain="science", outcome="held",
                             decisionRef="a", confidence=0.8)
    e2 = store.record_outcome(domain="science", outcome="contradicted",
                             decisionRef="b", confidence=0.8)
    assert isinstance(e1, SelfObservation) and isinstance(e2, SelfObservation)
    assert e2.prev_hash == e1.hash
    dumped = store.to_list()
    assert len(dumped) == 2
    assert dumped[0]["outcome"] == "held" and dumped[1]["outcome"] == "contradicted"
    assert dumped[0]["confidenceBand"] == "0.7-0.9"


def test_default_bands_shape():
    """DEFAULT_BANDS cover [0,1] contiguously (no gaps, no overlap)."""
    assert DEFAULT_BANDS[0][0] == 0.0
    assert DEFAULT_BANDS[-1][1] == 1.0
    for (lo1, hi1), (lo2, hi2) in zip(DEFAULT_BANDS, DEFAULT_BANDS[1:]):
        assert hi1 == lo2  # contiguous


if __name__ == "__main__":
    test_band_mapping()
    test_reliability_tracks_injected_truth()
    test_weak_prior_before_data()
    test_should_answer_defers_on_poor_track_record()
    test_compute_ece_matches_calibration_and_handworked()
    test_selective_risk_wrapper_and_ordering()
    test_store_level_ece_and_selective_risk()
    test_fail_closed_gate_rejects_ungated_write()
    test_default_gate_rejects_malformed_records()
    test_hash_chain_is_tamper_evident()
    test_records_chain_link_and_serialization()
    test_default_bands_shape()
    print("ALL TESTS PASSED")

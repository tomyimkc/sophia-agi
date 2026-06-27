# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifying tests for okf.decay_okf — the three honesty properties.

P1 no-silent-deletion: forgetting is demotion, never destruction.
P2 provenanced-forgetting: every suppression carries an auditable reason.
P3 source-discipline-outranks-recency: consensus is never time-decayed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf.decay_okf import BeliefState, DecayPlan, plan_decay, DEFAULT_HALF_LIFE_DAYS  # noqa: E402


def _belief(node_id, conf, age_days, surprise=0.0, reinforced=0, decayed=None):
    now = 1_700_000_000.0
    return BeliefState(
        node_id=node_id, author_confidence=conf,
        written_at=now - age_days * 86400, last_reinforced_at=now - age_days * 86400,
        surprise=surprise, reinforcement_count=reinforced, decayed_reason=decayed,
    )


def test_p1_no_silent_deletion_belief_count_is_non_decreasing():
    """Planning decay over N beliefs never emits a deletion (forgetting == demotion)."""
    beliefs = [_belief(f"n{i}", "attributed", age_days=i * 400) for i in range(8)]
    plan = plan_decay(beliefs, now=1_700_000_000.0)
    assert plan.deletions == 0
    assert plan.to_dict()["noSilentDeletion"] is True


def test_p2_every_suppression_has_a_provenanced_reason():
    """Every suppressed node carries a reason drawn from the controlled vocabulary."""
    beliefs = [
        _belief("old_weak", "legendary", age_days=2000),   # very decayed -> suppress
        _belief("fresh", "attributed", age_days=10),
    ]
    plan = plan_decay(beliefs, now=1_700_000_000.0)
    reasons = {n: r for n, r in plan.suppress}
    assert "old_weak" in reasons
    # reason must be from a controlled vocabulary — auditable, not free text
    head = reasons["old_weak"].split(":", 1)[0]
    assert head in {"time", "contradiction", "competition"}, reasons["old_weak"]


def test_p3_consensus_is_never_time_decayed():
    """A consensus belief, however old and unreinforced, is NOT suppressed by time."""
    consensus_old = _belief("c", "consensus", age_days=50_000)
    plan = plan_decay([consensus_old], now=1_700_000_000.0)
    suppressed = {n for n, _ in plan.suppress}
    assert "c" not in suppressed


def test_competition_suppresses_weak_tail_but_consensus_wins_outright():
    """In a contradiction group, the weak members lose to the strong — but a consensus
    claim wins outright regardless of recency, preserving source discipline."""
    weak_legend = _belief("legend", "legendary", age_days=10)        # fresh but weak
    strong_attributed = _belief("attr", "attributed", age_days=10)
    consensus_old = _belief("cons", "consensus", age_days=10_000)    # old but consensus
    plan = plan_decay(
        [weak_legend, strong_attributed, consensus_old], now=1_700_000_000.0,
        contradiction_groups={"q": ["legend", "attr", "cons"]},
    )
    suppressed = {n: r for n, r in plan.suppress}
    assert "cons" not in suppressed                  # consensus immune to competition too
    assert "legend" in suppressed                    # weakest loses
    assert suppressed["legend"].startswith("competition:")


def test_surprising_and_reinforced_beliefs_get_consolidated_not_decayed():
    """Surprise + reinforcement flips a belief toward consolidation, the engram signal."""
    surprising = _belief("novel", "attributed", age_days=5, surprise=0.9, reinforced=3)
    plan = plan_decay([surprising], now=1_700_000_000.0)
    assert "novel" in plan.reinforce


def test_tied_contradictions_are_quarantined_not_auto_resolved():
    """Two equal-strength contradictory claims must NOT be silently resolved."""
    a = _belief("a", "attributed", age_days=1)
    b = _belief("b", "attributed", age_days=1)       # identical strength -> tie
    plan = plan_decay([a, b], now=1_700_000_000.0, contradiction_groups={"q": ["a", "b"]})
    quarantined = {n for n, _ in plan.quarantine}
    assert "a" in quarantined and "b" in quarantined
    # and neither is auto-suppressed as a winner
    assert {n for n, _ in plan.suppress}.isdisjoint({"a", "b"})


def test_genesis_epoch_now_is_not_treated_as_unset():
    """now=0.0 (a GENESIS_EPOCH 'arrival unknown' marker) must be honoured, NOT swapped for
    the wall clock. Regression for the falsy-`now` bug that time-decayed the whole corpus
    on an UNMEASURED timestamp. With now==written_at, age is 0 -> no TIME suppression."""
    # base_rank>=1 belief at age 0: not suppressed at all.
    attributed = BeliefState(node_id="a", author_confidence="attributed",
                             written_at=0.0, last_reinforced_at=0.0)
    plan = plan_decay([attributed], now=0.0)
    assert plan.suppress == []
    # base_rank==0 belief IS suppressed at age 0 — but for low base confidence, NOT time.
    none_extant = BeliefState(node_id="z", author_confidence="none_extant",
                              written_at=0.0, last_reinforced_at=0.0)
    plan2 = plan_decay([none_extant], now=0.0)
    reasons = {n: r for n, r in plan2.suppress}
    assert "z" in reasons
    assert reasons["z"].split(":", 1)[0] == "epistemic_hygiene"   # not "time"


def test_effective_strength_exponential_decay_shape():
    """At one half-life, a non-consensus belief is at ~half base strength (sanity)."""
    b = _belief("x", "attributed", age_days=DEFAULT_HALF_LIFE_DAYS)
    s = b.effective_strength(1_700_000_000.0)
    # base_rank(attributed)=3; ~half after one half-life (within surprise floor of 1)
    assert 1.4 < s < 1.6


def main() -> int:
    test_p1_no_silent_deletion_belief_count_is_non_decreasing()
    test_p2_every_suppression_has_a_provenanced_reason()
    test_p3_consensus_is_never_time_decayed()
    test_competition_suppresses_weak_tail_but_consensus_wins_outright()
    test_surprising_and_reinforced_beliefs_get_consolidated_not_decayed()
    test_tied_contradictions_are_quarantined_not_auto_resolved()
    test_genesis_epoch_now_is_not_treated_as_unset()
    test_effective_strength_exponential_decay_shape()
    print("test_decay_okf: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

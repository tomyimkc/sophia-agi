# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifying tests for okf.frontier_demotion — the consensus-can-fall-but-only-decisively rule.

The thesis-level invariant under test: consensus is demotable in frontier domains, but
ONLY under K>=100 (decisive Bayes factor) AND N>=3 independent groups AND every obs
surprise-gated. Anything less -> quarantine, never demote. Demotion is exactly ONE rank.
"""
from __future__ import annotations

import math

from okf.frontier_demotion import (
    DemotionEvidence, Observation, try_demote_consensus,
    simulate_newton_to_einstein, simulate_opera_ftl_neutrino,
    K_DECISIVE, N_MIN,
)


# ---- Historical simulations: the headline falsifiable cases ----

def test_newton_to_einstein_regime_demotion():
    """The paradigm correct case: decisive evidence -> regime-scoped demotion, ONE rank."""
    sim = simulate_newton_to_einstein()
    assert sim["demoted"] is True
    assert sim["regimeScoped"] is True
    assert sim["newConfidence"] == "disputed"          # exactly ONE rank below consensus
    assert sim["stillConsensusInLowVelocity"] is True  # Newton not destroyed, regime-scoped


def test_opera_ftl_quarantined_by_multiplicity_floor():
    """The counter-case: one high-surprise event must NOT topple consensus."""
    sim = simulate_opera_ftl_neutrino()
    assert sim["quarantined"] is True
    assert sim["consensusUntouched"] is True
    # the reason must name the multiplicity floor
    assert any("multiplicity floor" in r for r in sim["reason"])


# ---- The three gates, each tested in isolation ----

def test_gate1_bayes_factor_must_be_decisive():
    """Strong evidence (K=50, below decisive=100) cannot demote — quarantine."""
    evidence = DemotionEvidence(
        bayes_factor=50.0,   # "very strong" per Jeffreys, but NOT decisive
        observations=tuple(Observation(f"o{i}", surprise_p=0.01, group=f"g{i}") for i in range(4)),
    )
    d = try_demote_consensus("consensus", evidence, domain_is_frontier=True)
    assert d.demote is False and d.quarantined is True
    assert any("decisive" in r for r in d.reasons)


def test_gate2_multiplicity_floor_blocks_single_experiment():
    """Even a decisive K with one observation group -> quarantine (the OPERA rule)."""
    evidence = DemotionEvidence(
        bayes_factor=1e6,                              # decisive
        observations=(Observation("solo", surprise_p=1e-9, group="only"),),  # N=1 < 3
    )
    d = try_demote_consensus("consensus", evidence, domain_is_frontier=True)
    assert d.demote is False and d.quarantined is True
    assert any("multiplicity floor" in r for r in d.reasons)


def test_gate3_surprise_filter_blocks_routine_confirmation():
    """A strong-K result that is NOT surprising under consensus cannot demote it
    (rules out manufactured demotions via routine confirmations)."""
    evidence = DemotionEvidence(
        bayes_factor=1e4,
        observations=(
            Observation("routine1", surprise_p=0.5, group="g1"),   # not surprising
            Observation("routine2", surprise_p=0.4, group="g2"),
            Observation("routine3", surprise_p=0.45, group="g3"),
        ),
    )
    d = try_demote_consensus("consensus", evidence, domain_is_frontier=True)
    assert d.demote is False and d.quarantined is True
    assert any("surprise gate" in r for r in d.reasons)


# ---- Source discipline preserved ----

def test_demotion_is_exactly_one_rank_never_to_provenance_categories():
    """The source-discipline guardrail: demotion is consensus->disputed, never to
    legendary/none_extant (provenance categories must not absorb evidence outcomes)."""
    evidence = DemotionEvidence(
        bayes_factor=1e6,
        observations=tuple(Observation(f"o{i}", surprise_p=0.01, group=f"g{i}") for i in range(4)),
        challenger_regime="new_regime",
    )
    d = try_demote_consensus("consensus", evidence, domain_is_frontier=True)
    assert d.demote is True
    assert d.new_confidence == "disputed"
    assert d.new_confidence not in {"legendary", "none_extant", "anachronism_risk"}
    assert d.to_dict()["rankDrop"] == 1
    assert d.to_dict()["neverToProvenanceCategories"] is True


def test_non_frontier_consensus_is_absolutely_immune():
    """P3 holds unconditionally for settled domains: even decisive evidence cannot demote."""
    evidence = DemotionEvidence(
        bayes_factor=1e9,
        observations=tuple(Observation(f"o{i}", surprise_p=0.001, group=f"g{i}") for i in range(5)),
    )
    d = try_demote_consensus("consensus", evidence, domain_is_frontier=False)
    assert d.demote is False and d.quarantined is False
    assert any("immune" in r for r in d.reasons)


def test_non_consensus_belief_skips_this_rule():
    """The rule only governs consensus demotion; other confidences are handled by decay_okf."""
    d = try_demote_consensus("attributed", DemotionEvidence(1e9, tuple()), domain_is_frontier=True)
    assert d.demote is False
    assert any("not consensus" in r for r in d.reasons)


# ---- Edge / adversarial ----

def test_shared_group_is_not_independent_multiplicity():
    """Adversarial: many observations from the same team/instrument count as ONE group."""
    evidence = DemotionEvidence(
        bayes_factor=1e6,
        observations=tuple(Observation(f"echo{i}", surprise_p=0.01, group="echo_chamber") for i in range(10)),
    )
    d = try_demote_consensus("consensus", evidence, domain_is_frontier=True)
    # 10 observations but ONE independence group -> floor not met -> quarantine
    assert d.demote is False and d.quarantined is True
    assert any("multiplicity floor" in r for r in d.reasons)


def test_global_demotion_when_challenger_has_no_regime():
    """If the challenger dominates everywhere (no regime), demotion is global single-rank."""
    evidence = DemotionEvidence(
        bayes_factor=1e6,
        observations=tuple(Observation(f"o{i}", surprise_p=0.01, group=f"g{i}") for i in range(4)),
        challenger_regime=None,   # no regime -> global
    )
    d = try_demote_consensus("consensus", evidence, domain_is_frontier=True)
    assert d.demote is True
    assert d.superseded_by_regime is None    # global
    assert d.new_confidence == "disputed"    # still ONE rank

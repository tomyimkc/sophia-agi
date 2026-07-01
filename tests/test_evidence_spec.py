# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for okf.evidence_spec: min-over-chain derivation + independence-collapse detection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from okf import evidence_spec as es
from okf import schema


def _spec():
    return es.load_spec()


def test_spec_loads_and_validates():
    spec = _spec()
    types = spec["evidenceTypes"]
    assert set(["citation", "formal-proof", "experimental", "verifier-pass",
                "witness-testimony", "consensus"]).issubset(types.keys())
    for t in types.values():
        assert t["confidenceCeiling"] in schema.AUTHOR_CONFIDENCE
        assert isinstance(t["minIndependentSources"], int)


def test_independence_report_collapses_shared_origin():
    sources = [
        {"id": "a", "origin": "farm_z"},
        {"id": "b", "origin": "farm_z"},
        {"id": "c", "origin": "farm_z"},
    ]
    rep = es.independence_report(sources)
    assert rep["rawSourceCount"] == 3
    assert rep["effectiveIndependentCount"] == 1, "three shared-origin sources must collapse to 1"
    assert rep["collapsed"] and rep["collapsed"][0]["count"] == 3


def test_independence_report_keeps_distinct_origins():
    sources = [{"id": "a", "origin": "r1"}, {"id": "b", "origin": "r2"}, {"id": "c", "origin": "r3"}]
    rep = es.independence_report(sources)
    assert rep["effectiveIndependentCount"] == 3
    assert rep["collapsed"] == []


def test_independence_origin_case_insensitive():
    sources = [{"id": "a", "origin": "Root"}, {"id": "b", "origin": "root"}, {"id": "c", "origin": "ROOT"}]
    rep = es.independence_report(sources)
    assert rep["effectiveIndependentCount"] == 1


def test_independence_missing_origin_is_self():
    # No declared origin -> each source is its own origin (does NOT silently collapse).
    sources = [{"id": "a"}, {"id": "b"}]
    rep = es.independence_report(sources)
    assert rep["effectiveIndependentCount"] == 2
    assert rep["collapsed"] == []


def test_min_over_chain_weakest_link_caps():
    spec = _spec()
    # A strong 'attributed' citation + a 'legendary' citation: chain caps at legendary (weakest).
    ev = [
        {"type": "citation", "confidence": "attributed", "sources": [{"id": "s1", "origin": "o1"}]},
        {"type": "citation", "confidence": "legendary", "sources": [{"id": "s2", "origin": "o2"}]},
    ]
    d = es.derive_confidence(ev, spec)
    assert d.derivedLabel == "legendary", "min-over-chain must not let a strong link launder a weak one"
    assert d.derivedRank == schema.confidence_rank("legendary")


def test_anachronism_risk_caps_at_zero():
    spec = _spec()
    ev = [
        {"type": "citation", "confidence": "attributed", "sources": [{"id": "s1", "origin": "o1"}]},
        {"type": "citation", "confidence": "anachronism_risk", "sources": [{"id": "s2", "origin": "o2"}]},
    ]
    d = es.derive_confidence(ev, spec)
    assert d.derivedRank == 0, "anachronism_risk (rank 0) must cap the whole chain"


def test_type_ceiling_caps_intrinsic():
    spec = _spec()
    # verifier-pass ceiling is 'attributed' — an intrinsic 'consensus' cannot exceed it.
    ev = [{"type": "verifier-pass", "confidence": "consensus",
           "sources": [{"id": "g", "origin": "gate1", "observedDate": "2026-06-01"}]}]
    d = es.derive_confidence(ev, spec, claimed="consensus", as_of="2026-07-01")
    assert d.derivedLabel == "attributed"
    assert d.inflated is True


def test_consensus_needs_three_independent_origins():
    spec = _spec()
    # Two distinct origins claiming consensus -> below the corroboration floor of 3.
    ev = [{"type": "consensus", "confidence": "consensus", "sources": [
        {"id": "r1", "origin": "rev_a", "observedDate": "2023-01-01"},
        {"id": "r2", "origin": "rev_b", "observedDate": "2023-06-01"},
    ]}]
    d = es.derive_confidence(ev, spec, claimed="consensus", as_of="2026-07-01")
    assert d.derivedLabel != "consensus", "2 origins cannot license consensus (floor=3)"
    assert d.inflated is True


def test_illusory_corroboration_collapses_below_floor():
    spec = _spec()
    # Three sources but ONE origin -> effective 1, cannot clear consensus floor.
    ev = [{"type": "consensus", "confidence": "consensus", "sources": [
        {"id": "a", "origin": "press", "observedDate": "2023-01-01"},
        {"id": "b", "origin": "press", "observedDate": "2023-02-01"},
        {"id": "c", "origin": "press", "observedDate": "2023-03-01"},
    ]}]
    d = es.derive_confidence(ev, spec, claimed="consensus", as_of="2026-07-01")
    assert d.effectiveIndependentCount == 1
    assert d.inflated is True
    assert d.collapsed, "the collapse must be reported"


def test_stale_experimental_fails_recency():
    spec = _spec()
    # One 28-year-old experiment: below minIndependentSources AND stale.
    ev = [{"type": "experimental", "confidence": "consensus", "sources": [
        {"id": "old", "origin": "lab", "observedDate": "1998-01-01"}]}]
    d = es.derive_confidence(ev, spec, claimed="consensus", as_of="2026-07-01")
    assert d.derivedRank < schema.confidence_rank("consensus")
    assert d.inflated is True


def test_empty_evidence_is_none_extant():
    spec = _spec()
    d = es.derive_confidence([], spec, claimed="attributed")
    assert d.derivedRank == 0
    assert d.derivedLabel == "none_extant"
    assert d.inflated is True


def test_honest_claim_not_inflated():
    spec = _spec()
    ev = [{"type": "citation", "confidence": "attributed",
           "sources": [{"id": "ed", "origin": "oxford"}]}]
    d = es.derive_confidence(ev, spec, claimed="attributed")
    assert d.inflated is False
    assert d.derivedLabel == "attributed"


def test_honest_consensus_three_origins_accepts():
    spec = _spec()
    ev = [{"type": "consensus", "confidence": "consensus", "sources": [
        {"id": "a", "origin": "rev_a", "observedDate": "2022-01-01"},
        {"id": "b", "origin": "rev_b", "observedDate": "2023-01-01"},
        {"id": "c", "origin": "rev_c", "observedDate": "2024-01-01"},
    ]}]
    d = es.derive_confidence(ev, spec, claimed="consensus", as_of="2026-07-01")
    assert d.derivedLabel == "consensus"
    assert d.inflated is False


def test_unknown_type_licenses_nothing():
    spec = _spec()
    ev = [{"type": "vibes", "confidence": "attributed", "sources": [{"id": "x", "origin": "y"}]}]
    d = es.derive_confidence(ev, spec, claimed="attributed")
    assert d.derivedRank == 0
    assert d.inflated is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL TESTS PASSED")

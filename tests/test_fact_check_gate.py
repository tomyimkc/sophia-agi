#!/usr/bin/env python3
"""Offline tests for the out-of-wiki fact-check gate."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_gate import (  # noqa: E402
    AtomicClaim,
    EvidenceSource,
    consensus_by_verification,
    decompose_and_type,
    external_ground,
    fact_check_text,
)


def test_decomposes_and_types_economic_claims() -> None:
    claims = decompose_and_type("GDP rose because subsidies changed incentives. 2 + 2 = 4.")
    types = [c.type for c in claims]
    assert "econ_causal" in types
    assert "math" in types


def test_deterministic_math_accepts_and_rejects() -> None:
    assert fact_check_text("2 + 2 = 4.").verdict == "accepted"
    bad = fact_check_text("2 + 2 = 5.")
    assert bad.verdict == "rejected"
    assert "computed 4" in bad.claims[0].reason


def test_url_holds_offline_then_accepts_with_resolver() -> None:
    offline = fact_check_text("The source is https://example.com/report.pdf.")
    assert offline.verdict == "held"
    online = fact_check_text("The source is https://example.com/report.pdf.", url_resolver=lambda url: True)
    assert online.verdict == "accepted"


def test_external_grounding_requires_independent_sources_for_high_risk() -> None:
    claim = AtomicClaim("Inflation rose because energy prices increased in 2022", "econ_causal", risk="high")

    def retriever(_claim):
        return [
            EvidenceSource(id="a", url="https://imf.org/a", title="Inflation rose because energy prices increased in 2022"),
            EvidenceSource(id="b", url="https://imf.org/b", title="Inflation rose because energy prices increased in 2022"),
        ]

    held = external_ground(claim, retriever)
    assert held.verdict == "held"  # same domain counts once; high risk requires 3 independent domains

    def retriever3(_claim):
        return [
            EvidenceSource(id="a", url="https://imf.org/a", title="Inflation rose because energy prices increased in 2022"),
            EvidenceSource(id="b", url="https://worldbank.org/b", title="Inflation rose because energy prices increased in 2022"),
            EvidenceSource(id="c", url="https://oecd.org/c", title="Inflation rose because energy prices increased in 2022"),
        ]

    accepted = external_ground(claim, retriever3)
    assert accepted.verdict == "accepted"


def test_external_grounding_rejects_contradiction() -> None:
    claim = AtomicClaim("GDP increased in 2020", "econ_empirical", risk="high")

    def retriever(_claim):
        return [EvidenceSource(id="a", url="https://stats.gov/a", title="GDP did not increase in 2020; the claim is false")]

    res = external_ground(claim, retriever)
    assert res.verdict == "rejected"


def test_consensus_is_not_vote_without_evidence_or_competence() -> None:
    claim = AtomicClaim("Regulatory capture caused the policy outcome", "econ_causal", risk="high")

    def rubber(_claim, _evidence):
        return {"family": "rubber", "verdict": "supports", "evidenceIds": [], "calibrationEce": 0.01, "rubberStampRate": 1.0, "heldoutN": 100}

    res = consensus_by_verification(claim, [rubber, rubber])
    assert res.verdict == "held"
    assert "majority vote is insufficient" in res.reason


def test_consensus_accepts_two_competent_families_with_evidence() -> None:
    claim = AtomicClaim("Regulatory capture caused the policy outcome", "econ_causal", risk="high")

    def j1(_claim, _evidence):
        return {"family": "anthropic", "verdict": "supports", "evidenceIds": ["s1"], "calibrationEce": 0.08, "rubberStampRate": 0.55, "heldoutN": 50}

    def j2(_claim, _evidence):
        return {"family": "openai", "verdict": "supports", "evidenceIds": ["s2"], "calibrationEce": 0.09, "rubberStampRate": 0.60, "heldoutN": 50}

    res = consensus_by_verification(claim, [j1, j2])
    assert res.verdict == "accepted"


def main() -> int:
    test_decomposes_and_types_economic_claims()
    test_deterministic_math_accepts_and_rejects()
    test_url_holds_offline_then_accepts_with_resolver()
    test_external_grounding_requires_independent_sources_for_high_risk()
    test_external_grounding_rejects_contradiction()
    test_consensus_is_not_vote_without_evidence_or_competence()
    test_consensus_accepts_two_competent_families_with_evidence()
    print("test_fact_check_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def test_subjective_meta_does_not_over_abstain() -> None:
    # Opinion/meta/question sentences are non-factual; they must not force a hold.
    d = fact_check_text("Let's think about this carefully. I recommend a cautious approach.")
    assert d.verdict == "accepted"
    assert all(c.claim.type == "subjective" for c in d.claims)
    q = fact_check_text("What should we do next?")
    assert q.verdict == "accepted"


def test_code_block_is_not_shattered() -> None:
    # Fence markers and code lines must not become bogus open_empirical claims.
    d = fact_check_text("Here is code:\n```python\nx = 1\ny = 2\nprint(x + y)\n```\n")
    types = [c.claim.type for c in d.claims]
    assert "code_python" in types
    assert not any(c.claim.text.strip().startswith("```") for c in d.claims)
    assert not any(c.claim.text.strip() in {"x = 1", "y = 2"} for c in d.claims)
    code = [c for c in d.claims if c.claim.type == "code_python"][0]
    assert code.verdict == "accepted"


def test_high_risk_lexical_screen_holds_below_floor() -> None:
    # Lexical overlap alone cannot pass a high-risk claim (entailment vs overlap).
    claim = AtomicClaim("Inflation rose because energy prices increased in 2022", "econ_causal", risk="high")

    def retr(_c):
        return [EvidenceSource(id=f"s{i}", url=f"https://{h}/x",
                               title="Inflation rose because energy prices increased in 2022")
                for i, h in enumerate(["imf.org", "worldbank.org", "oecd.org"], 1)]

    from agent.fact_check_gate import fact_check_claim
    dec = fact_check_claim(claim, retriever=retr)
    assert dec.verdict == "held"  # 0.78 lexical < 0.82 high-risk floor

    # A real entailment backend lifts it over the floor.
    dec2 = fact_check_claim(claim, retriever=retr, entailment=lambda c, s: "entails")
    assert dec2.verdict == "accepted"


def test_normal_risk_low_stakes_passes_on_lexical_screen() -> None:
    from agent.fact_check_gate import fact_check_claim
    claim = AtomicClaim("The library opened in 1998 according to records", "open_empirical", risk="normal")

    def retr(_c):
        return [EvidenceSource(id="a", url="https://a.org/x", title="The library opened in 1998 according to records"),
                EvidenceSource(id="b", url="https://b.org/y", title="records show the library opened in 1998")]

    dec = fact_check_claim(claim, retriever=retr)
    assert dec.verdict == "accepted"  # 0.78 >= 0.70 normal floor


def main() -> int:
    test_decomposes_and_types_economic_claims()
    test_deterministic_math_accepts_and_rejects()
    test_url_holds_offline_then_accepts_with_resolver()
    test_external_grounding_requires_independent_sources_for_high_risk()
    test_external_grounding_rejects_contradiction()
    test_consensus_is_not_vote_without_evidence_or_competence()
    test_consensus_accepts_two_competent_families_with_evidence()
    test_subjective_meta_does_not_over_abstain()
    test_code_block_is_not_shattered()
    test_high_risk_lexical_screen_holds_below_floor()
    test_normal_risk_low_stakes_passes_on_lexical_screen()
    print("test_fact_check_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

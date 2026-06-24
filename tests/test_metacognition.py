#!/usr/bin/env python3
"""Behaviour tests for the metacognition confidence heuristic (deterministic, offline).

This module is a confidence HEURISTIC (specificity + self-consistency + semantic
entropy + fact verdict + arithmetic), not introspection. These tests pin the
documented routing behaviour so the "what it literally is" claim stays honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.metacognition import (  # noqa: E402
    assess_uncertainty,
    normalize_answer,
    self_consistency,
    semantic_entropy,
)


def test_self_consistency_majority_and_agreement() -> None:
    ans, agree = self_consistency(["Paris", "paris", "Lyon"])
    assert ans == "paris" and abs(agree - 0.6667) < 1e-3
    assert self_consistency([]) == (None, 0.0)


def test_semantic_entropy_bounds() -> None:
    assert semantic_entropy(["same", "same", "same"]) == 0.0  # no disagreement
    assert semantic_entropy([]) == 1.0                        # nothing to go on
    assert semantic_entropy(["a", "b"]) > 0.0                 # split → entropy


def test_normalize_answer() -> None:
    assert normalize_answer("  The   ANSWER ") == "the answer"


def test_accepted_fact_allows() -> None:
    r = assess_uncertainty("Laozi is associated with the Dao De Jing.",
                           fact_verdict="accepted", fact_confidence=0.9, evidence_count=2)
    assert r.recommended_action == "allow"
    assert r.confidence >= 0.7


def test_rejected_fact_does_not_allow() -> None:
    r = assess_uncertainty("Roger Bacon wrote the Voynich manuscript.",
                           fact_verdict="rejected", evidence_count=0)
    assert r.recommended_action in {"abstain", "retrieve"}
    assert r.recommended_action != "allow"


def test_unsupported_specific_claim_does_not_allow() -> None:
    # A specific factual claim with no evidence must not sail through as "allow".
    r = assess_uncertainty("GDP rose 4.2% in 2021 because of policy X.",
                           evidence_count=0, high_risk=True)
    assert r.recommended_action != "allow"


def test_ambiguous_routes_to_clarify() -> None:
    r = assess_uncertainty("It depends — which one do you mean?")
    assert r.recommended_action == "clarify"
    assert r.uncertainty_type == "aleatoric"


def test_report_is_serializable() -> None:
    r = assess_uncertainty("A plain statement.", evidence_count=1)
    d = r.to_dict()
    for k in ("confidence", "uncertaintyType", "recommendedAction", "reasons"):
        assert k in d


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} metacognition tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

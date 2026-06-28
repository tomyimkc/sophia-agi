#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Concept-discipline RLVR reward (P7) — verifier-as-reward, bounded. Offline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.ontology_rl_reward import (  # noqa: E402
    REWARD_MIN,
    make_grpo_reward,
    ontology_rl_cases,
    reward_for_edge_case,
)


def test_floor_on_asserted_identity() -> None:
    r, d = reward_for_edge_case({"expected": "distinct"}, "Ren is identical to agape.")
    assert r == REWARD_MIN
    assert d["assertedForbidden"] is True


def test_grounded_distinction_wins() -> None:
    r, _ = reward_for_edge_case({"expected": "distinct"},
                                "No, ren and agape are distinct; they differ in tradition and grounding.")
    assert r == 1.0


def test_scoped_analogy_wins() -> None:
    r, _ = reward_for_edge_case({"expected": "scoped"},
                                "Wu wei resembles apatheia with respect to non-attached response.")
    assert r == 1.0


def test_bare_abstain_on_distinct_is_zero_not_negative() -> None:
    r, _ = reward_for_edge_case({"expected": "distinct"}, "I abstain; I cannot answer that.")
    assert r == 0.0


def test_over_refusal_penalized_on_admit() -> None:
    r, d = reward_for_edge_case({"expected": "admit"}, "I abstain; I cannot answer that.")
    assert r == -0.5
    assert d.get("overRefusal") is True


def test_verifier_seam_invoked() -> None:
    spy: dict = {}
    reward_for_edge_case({"expected": "distinct"}, "Ren is identical to agape.", spy=spy)
    assert spy["verifier_calls"] == 1


def test_cases_have_both_expected_types() -> None:
    cases = ontology_rl_cases()
    kinds = {c["expected"] for c in cases}
    assert "distinct" in kinds and "admit" in kinds
    assert len(cases) >= 10


def test_grpo_reward_factory_routes_by_column() -> None:
    fn = make_grpo_reward()
    completions = ["Ren is identical to agape.",
                   "No, ren and agape are distinct and differ."]
    rewards = fn(["p", "p"], completions, expected=["distinct", "distinct"], answerable=[True, True])
    assert rewards[0] == REWARD_MIN  # asserted forbidden identity
    assert rewards[1] == 1.0         # grounded distinction


def main() -> int:
    test_floor_on_asserted_identity()
    test_grounded_distinction_wins()
    test_scoped_analogy_wins()
    test_bare_abstain_on_distinct_is_zero_not_negative()
    test_over_refusal_penalized_on_admit()
    test_verifier_seam_invoked()
    test_cases_have_both_expected_types()
    test_grpo_reward_factory_routes_by_column()
    print("test_ontology_rl_reward: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

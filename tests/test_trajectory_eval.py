#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Agent-trajectory evaluator: per-step + whole-run faithfulness, fail-closed."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.trajectory_eval import (  # noqa: E402
    BLOCKED,
    GROUNDED,
    SKIPPED,
    UNGROUNDED,
    UNVERIFIED,
    Support,
    evaluate_trajectory,
    lexical_support_judge,
)


def _verdicts(result):
    return [s["verdict"] for s in result["steps"]]


def test_grounded_claim_on_own_observation_accepts() -> None:
    traj = [
        {"id": "s1", "role": "tool_call", "tool": "search",
         "observation": "The capital of France is Paris, a city on the Seine."},
        {"id": "s2", "role": "assertion",
         "claim": "The capital of France is Paris on the Seine."},
    ]
    res = evaluate_trajectory(traj)
    assert res["verdict"] == "accept", res["reasons"]
    assert res["faithfulnessScore"] == 1.0
    assert res["firstUnfaithfulStep"] is None
    assert _verdicts(res) == [SKIPPED, GROUNDED]


def test_claim_with_no_evidence_is_ungrounded_and_abstains() -> None:
    traj = [
        {"id": "s1", "role": "assertion",
         "claim": "Quarterly revenue grew forty-two percent year over year."},
    ]
    res = evaluate_trajectory(traj)
    assert res["verdict"] == "abstain", res["reasons"]
    assert res["ungroundedSteps"] == ["s1"]
    assert res["firstUnfaithfulStep"] == "s1"
    assert res["faithfulnessScore"] == 0.0


def test_forward_citation_earns_nothing() -> None:
    # s1 cites s2, which appears LATER -> not an earlier verified step -> ungrounded.
    traj = [
        {"id": "s1", "role": "assertion", "claim": "Widget sales doubled in Q3.",
         "cites": ["s2"]},
        {"id": "s2", "role": "observation",
         "observation": "Widget sales doubled in Q3 versus Q2."},
    ]
    res = evaluate_trajectory(traj)
    step1 = res["steps"][0]
    assert step1["verdict"] == UNGROUNDED
    assert any("not an earlier step" in p for p in step1["citationProblems"])


def test_valid_backward_citation_grounds() -> None:
    traj = [
        {"id": "s1", "role": "observation",
         "observation": "Patient temperature recorded at thirty-nine degrees celsius."},
        {"id": "s2", "role": "tool_call", "tool": "noop"},
        {"id": "s3", "role": "assertion",
         "claim": "Patient temperature was thirty-nine degrees celsius.",
         "cites": ["s1"]},
    ]
    res = evaluate_trajectory(traj)
    assert res["steps"][2]["verdict"] == GROUNDED
    assert res["verdict"] == "accept"


def test_provenance_violation_blocks_the_trajectory() -> None:
    traj = [
        {"id": "s1", "role": "observation",
         "observation": "Confucius and Laozi founded distinct traditions."},
        {"id": "s2", "role": "assertion",
         "claim": "Confucius wrote the Dao De Jing and merged Daoist and Confucian ideas.",
         "cites": ["s1"]},
    ]
    res = evaluate_trajectory(traj)
    assert res["verdict"] == "blocked", res["reasons"]
    assert res["blockedSteps"] == ["s2"]
    assert res["steps"][1]["verdict"] == BLOCKED
    assert res["steps"][1]["violations"]


def test_evidence_present_but_weak_overlap_abstains_not_condemns() -> None:
    # Evidence exists but lexically unrelated -> UNVERIFIED (abstain), never GROUNDED.
    traj = [
        {"id": "s1", "role": "observation", "observation": "Mitochondria produce ATP."},
        {"id": "s2", "role": "assertion",
         "claim": "The defendant breached the lease covenant in March.",
         "cites": ["s1"]},
    ]
    res = evaluate_trajectory(traj)
    assert res["steps"][1]["verdict"] == UNVERIFIED
    assert res["verdict"] == "abstain"


def test_no_claims_abstains_with_nothing_to_certify() -> None:
    traj = [
        {"id": "s1", "role": "tool_call", "tool": "search", "observation": "some text"},
        {"id": "s2", "role": "tool_call", "tool": "open"},
    ]
    res = evaluate_trajectory(traj)
    assert res["verdict"] == "abstain"
    assert res["faithfulnessScore"] is None
    assert "nothing to certify" in " ".join(res["reasons"])


def test_injected_judge_resolves_unverified_to_grounded() -> None:
    # An entailment judge that always supports turns the weak-overlap abstain into
    # an accept — proving the judge is actually consulted and is pluggable.
    def always_support(claim: str, evidence: str) -> Support:
        return Support(supported=True, abstained=False, reason="stub", method="stub")

    traj = [
        {"id": "s1", "role": "observation", "observation": "Mitochondria produce ATP."},
        {"id": "s2", "role": "assertion",
         "claim": "The defendant breached the lease covenant.", "cites": ["s1"]},
    ]
    res = evaluate_trajectory(traj, judge=always_support)
    assert res["steps"][1]["verdict"] == GROUNDED
    assert res["verdict"] == "accept"


def test_broken_judge_abstains_does_not_certify() -> None:
    def boom(claim: str, evidence: str) -> Support:
        raise RuntimeError("judge exploded")

    traj = [
        {"id": "s1", "observation": "Aspirin can reduce fever."},
        {"id": "s2", "claim": "Aspirin can reduce fever in adults.", "cites": ["s1"]},
    ]
    res = evaluate_trajectory(traj, judge=boom)
    # Fail-closed: a broken judge can never produce GROUNDED.
    assert res["steps"][1]["verdict"] == UNVERIFIED
    assert res["verdict"] == "abstain"


def test_default_judge_is_lexical_and_deterministic() -> None:
    judge = lexical_support_judge()
    s = judge("Paris is the capital of France", "The capital of France is Paris.")
    assert s.supported and not s.abstained
    s2 = judge("Paris is the capital of France", "Mitochondria produce ATP.")
    assert s2.abstained and not s2.supported


def test_non_dict_step_is_skipped_safely() -> None:
    res = evaluate_trajectory([None, "oops", {"claim": "x", "observation": "x y z"}])
    assert res["steps"][0]["verdict"] == SKIPPED
    assert res["steps"][1]["verdict"] == SKIPPED


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

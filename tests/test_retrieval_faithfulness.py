#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the retrieval-faithfulness RLVR reward. Offline, no torch.

Asserts the invariants the offline claim rests on (determinism, bounded,
verifier-seam-traversed) plus the constructs that make this reward DIFFERENT
from rl_reward: the counterfactual citation-drop faithfulness term, the
contradicted / fabricated-citation hard floors, the confidence-laundering
provenance term, the when-to-retrieve decision term, and the cost / hedge / over-
refusal guards. The live GRPO uplift claim is deliberately NOT tested here — it
stays Open until a pre-registered, powered, multi-family run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import retrieval_faithfulness as rf  # noqa: E402

R = rf.reward_for_trajectory


def _claim(**kw):
    base = {
        "text": "X was written by A.",
        "kind": "knowledge",
        "verdict": "supported",
        "support_chunk_ids": ["c1"],
        "support_confidences": ["attributed"],
        "survives_ablation": False,
        "asserted_confidence": "attributed",
    }
    base.update(kw)
    return base


def _traj(**kw):
    base = {
        "task_correct": True,
        "claims": [_claim()],
        "retrieved_ids": ["c1"],
        "context_ids": ["c1"],
        "should_retrieve": True,
        "did_retrieve": True,
        "n_retrievals": 1,
        "abstained": False,
        "answerable": True,
        "answer_text": "X was written by A, per the retrieved record.",
    }
    base.update(kw)
    return base


# --- core invariants ------------------------------------------------------- #


def test_reward_is_deterministic() -> None:
    assert R(_traj())[0] == R(_traj())[0]


def test_reward_is_bounded() -> None:
    trajs = [
        _traj(),
        _traj(task_correct=False, claims=[_claim(verdict="unsupported")]),
        _traj(claims=[_claim(verdict="contradicted")]),
        _traj(abstained=True, answerable=False, claims=[]),
        _traj(claims=[]),
    ]
    for t in trajs:
        r, _ = R(t)
        assert rf.REWARD_MIN <= r <= rf.REWARD_MAX


def test_reward_traverses_claim_seam() -> None:
    spy: dict = {}
    R(_traj(claims=[_claim(), _claim(text="Y by B.")]), spy=spy)
    assert spy["claims_scored"] == 2  # both knowledge claims were actually scored


# --- the faithfulness centerpiece ------------------------------------------ #


def test_surviving_claim_scores_below_flipping_claim() -> None:
    """A claim that SURVIVES dropping its support leaked from the weights and must
    score below an otherwise-identical claim that FLIPS (genuinely grounded)."""
    grounded = R(_traj(claims=[_claim(survives_ablation=False)]))[0]
    leaked = R(_traj(claims=[_claim(survives_ablation=True)]))[0]
    assert grounded > leaked


def test_commonsense_claim_not_required_to_be_grounded() -> None:
    """A commonsense/reasoning step is excluded from grounding+faithfulness, so a
    rollout with one grounded knowledge claim + one commonsense step is not
    penalized for the latter being ungrounded."""
    t = _traj(claims=[_claim(), {"text": "2+2=4", "kind": "commonsense",
                                 "verdict": "unsupported", "support_chunk_ids": []}])
    r, detail = R(t)
    assert detail["nKnowledge"] == 1
    assert r > 0.5


# --- hard floors ----------------------------------------------------------- #


def test_contradicted_claim_is_hard_floor() -> None:
    r, detail = R(_traj(claims=[_claim(verdict="contradicted")]))
    assert r == rf.REWARD_MIN
    assert detail.get("assertedContradicted") is True


def test_fabricated_citation_is_hard_floor() -> None:
    """Citing a chunk that was never retrieved is fabrication -> hard floor."""
    r, detail = R(_traj(claims=[_claim(support_chunk_ids=["ghost"])],
                        retrieved_ids=["c1"], context_ids=["c1"]))
    assert r == rf.REWARD_MIN
    assert detail.get("fabricatedCitation") is True


# --- provenance: no confidence laundering ---------------------------------- #


def test_confidence_laundering_penalized() -> None:
    """Asserting consensus-level certainty over a legendary-confidence source is
    laundering; it must score below an honestly-hedged equivalent."""
    honest = R(_traj(claims=[_claim(asserted_confidence="legendary",
                                    support_confidences=["legendary"])]))[0]
    laundered = R(_traj(claims=[_claim(asserted_confidence="consensus",
                                       support_confidences=["legendary"])]))[0]
    assert honest > laundered


# --- when-to-retrieve decision --------------------------------------------- #


def test_not_retrieving_when_needed_is_penalized() -> None:
    answered = R(_traj(should_retrieve=True, did_retrieve=True))[0]
    skipped = R(_traj(should_retrieve=True, did_retrieve=False, n_retrievals=0))[0]
    assert answered > skipped


def test_over_retrieval_costs() -> None:
    one = R(_traj(should_retrieve=True, n_retrievals=1))[0]
    many = R(_traj(should_retrieve=True, n_retrievals=6))[0]
    assert one > many


# --- abstention: fail-closed is good, over-refusal is not ------------------ #


def test_correct_abstention_rewarded_overrefusal_penalized() -> None:
    good = R(_traj(abstained=True, answerable=False, claims=[]))[0]
    miss = R(_traj(abstained=True, answerable=True, claims=[]))[0]
    assert good > 0.0 > miss


# --- anti-hedging cap (shared with rl_reward) ------------------------------ #


def test_anti_hedging_cap() -> None:
    hedgy = (
        "This is traditionally and commonly attributed, though disputed, debated, "
        "and apocryphal."
    )
    r, detail = R(_traj(answer_text=hedgy))
    assert detail["hedges"] > rf._MAX_HEDGES
    assert r <= 0.4
    assert detail.get("hedgingCapped") is True


def main() -> int:
    test_reward_is_deterministic()
    test_reward_is_bounded()
    test_reward_traverses_claim_seam()
    test_surviving_claim_scores_below_flipping_claim()
    test_commonsense_claim_not_required_to_be_grounded()
    test_contradicted_claim_is_hard_floor()
    test_fabricated_citation_is_hard_floor()
    test_confidence_laundering_penalized()
    test_not_retrieving_when_needed_is_penalized()
    test_over_retrieval_costs()
    test_correct_abstention_rewarded_overrefusal_penalized()
    test_anti_hedging_cap()
    print("test_retrieval_faithfulness: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

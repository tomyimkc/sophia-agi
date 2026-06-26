#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the image-grounded hallucination-trap suite (multimodal_bench).

Covers Phase 1 of the multimodal roadmap end to end, fully offline:
* dataset integrity — every trap's human-readable gold equals its judge-free
  verifier (labels are machine-derived, never hand-asserted);
* the verifiers (count / presence / relation / OCR) compute correctly;
* the lexical judge parses yes-no / count / text answers and abstentions;
* multi-judge consensus majority-votes and reports kappa;
* the gate end-to-end: a grounded model hallucinates 0% and grounds 100%, a
  credulous model hallucinates on every trap row, an abstainer abstains;
* aggregation attaches bootstrap CIs and the no-overclaim validation flags
  (a mock run is never 'validated').
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_bench import judge as J
from multimodal_bench import model, runner, verifiers


# --- dataset integrity ----------------------------------------------------- #

def test_at_least_30_traps_with_all_categories():
    traps = runner.load_traps()
    assert len(traps) >= 30
    cats = {t["category"] for t in traps}
    for required in ("phantom_object", "miscount", "spatial_relation", "fabricated_ocr"):
        assert required in cats
    # discrimination controls exist so the gate can't win by blanket-denying
    assert "presence_control" in cats and "spatial_control" in cats


def test_trap_ids_unique_and_fields_present():
    traps = runner.load_traps()
    ids = [t["id"] for t in traps]
    assert len(ids) == len(set(ids))
    for t in traps:
        for f in ("scene", "question", "answer_type", "gold_answer", "trap_answer", "check", "reason"):
            assert f in t, f"{t['id']} missing {f}"
        assert t["answer_type"] in ("yesno", "count", "text")
        assert str(t["gold_answer"]).lower() != str(t["trap_answer"]).lower()


def test_every_gold_label_matches_its_verifier():
    """The dataset invariant: machine-derived label == human-readable gold."""
    for t in runner.load_traps():
        assert verifiers.gold_matches_check(t), f"label/verifier mismatch on {t['id']}"


# --- verifiers ------------------------------------------------------------- #

def test_count_and_presence():
    scene = {"objects": [{"label": "apple", "box": [0, 0, 10, 10]},
                         {"label": "apple", "box": [20, 0, 10, 10]},
                         {"label": "pear", "box": [40, 0, 10, 10]}], "texts": []}
    assert verifiers.count(scene, "apple") == 2
    assert verifiers.present(scene, "apple") is True
    assert verifiers.present(scene, "banana") is False


def test_relation_semantics():
    scene = {"objects": [{"label": "dog", "box": [0, 200, 50, 50]},
                         {"label": "cat", "box": [300, 200, 50, 50]}], "texts": []}
    assert verifiers.relation(scene, "dog", "left_of", "cat") is True
    assert verifiers.relation(scene, "cat", "left_of", "dog") is False
    # y grows downward: smaller y == above
    scene2 = {"objects": [{"label": "bird", "box": [0, 10, 20, 20]},
                          {"label": "cloud", "box": [0, 300, 40, 40]}], "texts": []}
    assert verifiers.relation(scene2, "bird", "above", "cloud") is True
    assert verifiers.relation(scene2, "bird", "below", "cloud") is False
    # missing object -> False, never an exception
    assert verifiers.relation(scene2, "bird", "above", "missing") is False


def test_ocr_contains_and_blank_sentinel():
    scene = {"objects": [], "texts": [{"value": "OPEN", "box": [0, 0, 10, 10]}]}
    assert verifiers.ocr_contains(scene, "open") is True       # case/format tolerant
    assert verifiers.ocr_contains(scene, "CLOSED") is False
    assert verifiers.ocr_contains(scene, "none") is False      # text IS present
    blank = {"objects": [], "texts": []}
    assert verifiers.ocr_contains(blank, "none") is True       # honest 'no text'


# --- judge ----------------------------------------------------------------- #

def test_lexical_judge_yesno():
    trap = {"answer_type": "yesno", "gold_answer": "no", "trap_answer": "yes"}
    assert J.lexical_judge("No, there is no cat.", trap).affirmed_gold is True
    assert J.lexical_judge("Yes, a cat is on the sofa.", trap).hallucinated is True
    assert J.lexical_judge("I can't tell from this image.", trap).abstained is True


def test_lexical_judge_count_and_text():
    ctrap = {"answer_type": "count", "gold_answer": "3", "trap_answer": "4"}
    assert J.lexical_judge("There are three apples.", ctrap).affirmed_gold is True
    assert J.lexical_judge("I count 4 apples.", ctrap).hallucinated is True
    ttrap = {"answer_type": "text", "gold_answer": "OPEN", "trap_answer": "CLOSED"}
    assert J.lexical_judge("The sign says OPEN.", ttrap).affirmed_gold is True
    assert J.lexical_judge("It reads CLOSED.", ttrap).hallucinated is True
    blank = {"answer_type": "text", "gold_answer": "none", "trap_answer": "SALE 50% OFF"}
    assert J.lexical_judge("There is no text on the sign.", blank).affirmed_gold is True
    assert J.lexical_judge("It says SALE 50% OFF.", blank).hallucinated is True


# --- consensus ------------------------------------------------------------- #

def test_consensus_majority_and_agreement():
    def yes_judge(a, t):  # always votes hallucinated
        return J.VisualJudgment(abstained=False, hallucinated=True, affirmed_gold=False)

    def no_judge(a, t):   # never votes hallucinated
        return J.VisualJudgment(abstained=False, hallucinated=False, affirmed_gold=True)

    # 2 hallucinate vs 1 not -> majority hallucinated
    cj = J.make_consensus_judge([yes_judge, yes_judge, no_judge], labels=["a:x", "a:y", "b:z"])
    v = cj("whatever", {"answer_type": "yesno", "gold_answer": "no", "trap_answer": "yes"})
    assert v.hallucinated is True and v.method == "consensus:3"
    assert len(v.votes) == 3
    with_votes = [v.votes]
    agree = J.percent_agreement(with_votes)
    assert agree is not None and "meanPairwiseKappa" in agree


def test_consensus_requires_two_judges():
    try:
        J.make_consensus_judge([J.lexical_judge])
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- gate end-to-end ------------------------------------------------------- #

def test_grounded_model_never_hallucinates():
    traps = runner.load_traps()
    rows = runner.run_cases(traps, model.grounded_answer_fn)
    assert all(not r["hallucinated"] for r in rows)
    # a perfect honest model should ground (affirm gold) on essentially all rows
    grounded = sum(r["affirmed_gold"] for r in rows)
    assert grounded >= int(0.95 * len(rows))


def test_credulous_model_hallucinates_on_trap_rows():
    traps = runner.load_traps()
    rows = runner.run_cases(traps, model.credulous_answer_fn)
    # every row has a documented-false trap answer; the credulous model takes it
    halluc = sum(r["hallucinated"] for r in rows)
    assert halluc >= int(0.95 * len(rows))


def test_abstainer_model_abstains():
    traps = runner.load_traps()
    rows = runner.run_cases(traps, model.abstainer_answer_fn)
    assert all(r["abstained"] for r in rows)
    assert all(not r["hallucinated"] for r in rows)


# --- aggregation + no-overclaim ------------------------------------------- #

def test_aggregate_has_ci_and_blocks_mock_headline():
    traps = runner.load_traps()
    runs = [runner.run_cases(traps, model.credulous_answer_fn) for _ in range(3)]
    rep = runner.aggregate_runs(runs, model_spec="mock:credulous")
    assert rep["hallucinationRate"] > 0.9
    assert len(rep["ciHallucination"]) == 2
    assert rep["ciHallucination"][0] <= rep["hallucinationRate"] <= rep["ciHallucination"][1]
    assert "byCategory" in rep and "miscount" in rep["byCategory"]
    # mock runs can never be a validated headline
    assert rep["validated"] is False
    assert rep["validatedChecks"]["notMock"] is False


def test_resolve_answer_fn_specs():
    assert model.resolve_answer_fn("mock:grounded") is model.grounded_answer_fn
    for bad in ("mock:nope", "weird:thing"):
        try:
            model.resolve_answer_fn(bad)
            assert False, f"expected ValueError for {bad}"
        except ValueError:
            pass


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for multimodal roadmap phases 2-3, fully offline.

Phase 2 (RLVR): the visual-grounding reward is verifier-derived, bounded, honest
(correct > abstain > wrong), fail-closed on a verifier mismatch, TRL-shaped, and
backed by a contamination-free family split.

Phase 3 (data synthesis): the chart/table/document traps regenerate
deterministically and every synthetic label is re-derivable by the verifier.

Phase 3 (calibration): risk-coverage over the suite separates a calibrated VLM
(selective risk < base risk, low AURC) from an overconfident one.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_bench import (
    calibration,
    runner,
    synthesize,
    verifiers,
    visual_dataset,
    visual_reward,
)


# --- Phase 2: verifier-as-reward ------------------------------------------ #

def test_reward_ordering_and_bounds():
    ok, detail = visual_reward.offline_invariants()
    assert ok, detail["checks"]
    c = detail["checks"]
    assert c["correctPositive"] and c["wrongNegative"] and c["honestyOrdering"]
    assert c["verifierSeamInvoked"] and c["bounded"] and c["contaminationFree"]


def test_reward_fail_closed_on_verifier_mismatch():
    # A trap whose declared gold contradicts the scene -> held at wrong reward.
    bad = {
        "id": "bad", "category": "phantom_object", "answer_type": "yesno",
        "gold_answer": "yes", "trap_answer": "no",  # claims a cat IS present...
        "scene": {"objects": [{"label": "sofa", "box": [0, 0, 10, 10]}], "texts": []},
        "check": {"type": "presence", "label": "cat", "expect": True},  # ...but none is
    }
    assert verifiers.gold_matches_check(bad) is False
    score, detail = visual_reward.reward_for_trap("Yes, a cat.", bad)
    assert detail["category"] == "held" and detail["reason"] == "verifier_mismatch"
    assert score == visual_reward.REWARD_MIN


def test_reward_prefers_abstention_over_hallucination():
    trap = {
        "id": "t", "category": "phantom_object", "answer_type": "yesno",
        "gold_answer": "no", "trap_answer": "yes",
        "scene": {"objects": [{"label": "sofa", "box": [0, 0, 10, 10]}], "texts": []},
        "check": {"type": "presence", "label": "cat", "expect": False},
    }
    r_correct, _ = visual_reward.reward_for_trap("No, there is no cat.", trap)
    r_abstain, _ = visual_reward.reward_for_trap("I can't tell.", trap)
    r_wrong, _ = visual_reward.reward_for_trap("Yes, a cat.", trap)
    assert r_correct > r_abstain > r_wrong


def test_grpo_reward_signature_and_values():
    traps = runner.load_all_traps()[:4]
    reward_fn = visual_reward.make_grpo_reward()
    # a grounded completion per trap -> all correct (+1)
    from multimodal_bench import model
    completions = [model.grounded_answer_fn(t) for t in traps]
    scores = reward_fn(["p"] * len(traps), completions, trap=traps)
    assert scores == [visual_reward.REWARD_MAX] * len(traps)


def test_rl_split_is_family_disjoint():
    data = visual_dataset.build_visual_rl_dataset(eval_frac=0.34, seed=0)
    assert data["family_intersection"] == []
    assert set(data["train_families"]).isdisjoint(data["eval_families"])
    assert data["train_rows"] and data["eval_rows"]


# --- Phase 3: verifier-checked synthesis ---------------------------------- #

def test_synth_is_deterministic():
    a = synthesize.build_synth_traps(seed=0)
    b = synthesize.build_synth_traps(seed=0)
    assert a == b
    assert {t["category"] for t in a} == {"chart_qa", "table_qa", "document_qa"}


def test_committed_synth_file_matches_generator():
    """The frozen data file must equal a fresh generation (no silent drift)."""
    on_disk = runner.load_traps(runner.SYNTH_PATH)
    fresh = synthesize.build_synth_traps(seed=0)
    assert on_disk == fresh, "data/visual_traps_synth.json is stale; regenerate it"


def test_every_synth_label_matches_verifier():
    for t in synthesize.build_synth_traps(seed=0):
        assert verifiers.gold_matches_check(t), f"synth label/verifier mismatch on {t['id']}"


def test_chart_table_doc_verifiers():
    scene = {"chart": {"bars": [{"label": "A", "value": 10}, {"label": "B", "value": 40}, {"label": "C", "value": 25}]},
             "table": {"columns": ["item", "price"], "rows": [["apple", "3"], ["pear", "7"]]},
             "document": {"fields": {"Total": "$12.99", "Due": "Net 30"}}}
    assert verifiers.chart_value(scene, "B") == 40
    assert verifiers.chart_extreme(scene, "max") == "B"
    assert verifiers.chart_extreme(scene, "min") == "A"
    assert verifiers.table_cell(scene, "pear", "price") == "7"
    assert verifiers.table_cell(scene, "missing", "price") is None
    assert verifiers.doc_field(scene, "Total") == "$12.99"


def test_load_all_includes_synth():
    base = runner.load_traps()
    allt = runner.load_all_traps()
    assert len(allt) == len(base) + len(synthesize.build_synth_traps(seed=0))


# --- Phase 3: calibration / risk-coverage --------------------------------- #

def test_calibrated_beats_overconfident_on_risk_coverage():
    out = calibration.demo()
    cal_rep, over_rep = out["calibrated"], out["overconfident"]
    # calibrated: thresholding cuts risk and AURC is low
    assert cal_rep["selectiveRisk"]["0.5"] < cal_rep["baseRisk"]
    assert cal_rep["aurc"] < over_rep["aurc"]
    # overconfident: high-confidence prefix is no safer than the whole set
    assert over_rep["selectiveRisk"]["0.5"] >= cal_rep["selectiveRisk"]["0.5"]
    # full coverage (1.0) risk equals base risk for both
    assert cal_rep["selectiveRisk"]["1.0"] == cal_rep["baseRisk"]


def test_calibration_report_shape():
    traps = runner.load_all_traps()
    fn = calibration.make_synthetic_confidence_fn(error_rate=0.3, calibrated=True, seed=1)
    rep = calibration.calibration_report(calibration.run_with_confidence(traps, fn))
    for k in ("ece", "baseRisk", "aurc", "selectiveRisk", "riskCoverageCurve"):
        assert k in rep
    assert 0.0 <= rep["ece"] <= 1.0
    assert len(rep["riskCoverageCurve"]) == len(traps)


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

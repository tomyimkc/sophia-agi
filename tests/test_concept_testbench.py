# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the concept-discipline testbench (metrics, A/B, ablation, RL).

These are the falsifiable invariants the testbench rests on: the metrics compute
the asymmetric quantities correctly, the inference A/B shows a violation-rate drop
whose bootstrap CI excludes 0 when the gate is applied to a careless policy, the
spurious-reward ablation discriminates (true signal separates, random does not),
the RL split is contamination-free, and the concept reward machinery is sound.
All run on Apple Silicon / CI without a GPU or a model."""
from __future__ import annotations

import json
from pathlib import Path

from provenance_bench import (
    concept_metrics,
    ontology_improvement,
    ontology_rl_dataset,
    ontology_rl_reward,
    spurious_ablation,
)

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval" / "philosopher_reasoning" / "philosopher_reasoning_v1.jsonl"


def _items() -> list[dict]:
    return [json.loads(l) for l in EVAL.read_text(encoding="utf-8").splitlines() if l.strip()]


# --- B1 metrics -------------------------------------------------------------- #
def test_summarize_counts_violation_and_abstention():
    recs = [
        {"answerable": True, "abstained": False, "violation": True, "correct": False},
        {"answerable": True, "abstained": False, "violation": False, "correct": True},
        {"answerable": False, "abstained": True, "violation": False, "correct": True},
    ]
    s = concept_metrics.summarize(recs)
    assert s["conceptMergeViolationRate"] == 1 / 3
    assert s["abstentionRecall"] == 1.0  # the one ill-posed item was abstained
    assert s["confidentWrongRate"] == 0.5  # 1 of 2 answerable answered wrong, none abstained


def test_bootstrap_delta_excludes_zero_for_clear_separation():
    a = [0.0] * 20
    b = [1.0] * 20
    d = concept_metrics.bootstrap_delta(a, b, seed=0)
    assert d["delta"] == 1.0 and d["excludesZero"] is True


def test_bootstrap_delta_includes_zero_for_no_signal():
    a = [0.0, 1.0] * 10
    b = [0.0, 1.0] * 10
    d = concept_metrics.bootstrap_delta(a, b, seed=0)
    assert d["excludesZero"] is False


# --- B2 inference A/B -------------------------------------------------------- #
def test_guard_blocks_merge_and_passes_benign():
    assert ontology_improvement.guard("Ren is identical to agape.") == ontology_improvement.GUARDED_ABSTENTION
    assert ontology_improvement.guard("The sky is blue.") == "The sky is blue."


def test_inference_uplift_violation_rate_drops_with_ci():
    items = _items()
    report = ontology_improvement.run_ab(items, ontology_improvement.naive_policy,
                                         repair=ontology_improvement.reference_repair, seed=0)
    base = report["baseline"]["conceptMergeViolationRate"]
    treat = report["treatment"]["conceptMergeViolationRate"]
    delta = report["deltas"]["conceptMergeViolationRate"]
    assert base > 0.0  # the careless policy DOES merge cross-tradition concepts
    assert treat == 0.0  # the gate removes every merge
    assert delta["delta"] < 0 and delta["excludesZero"]  # drop, CI excludes 0


def test_guard_alone_trades_violations_for_abstentions():
    # Without a repair, the gate converts a careless merge into an abstention: the
    # honest tradeoff the over-abstention tripwire is designed to catch.
    items = _items()
    base = ontology_improvement.run_arm(items, ontology_improvement.naive_policy, guarded=False)
    treat = ontology_improvement.run_arm(items, ontology_improvement.naive_policy, guarded=True)  # no repair
    trip = spurious_ablation.over_abstention_tripwire(base, treat)
    assert trip["tripped"] is True


# --- B4 spurious ablation ---------------------------------------------------- #
def test_spurious_ablation_discriminates():
    abl = spurious_ablation.run_spurious_ablation(seed=0)
    assert abl["trueRewardDelta"]["excludesZero"] is True
    assert abl["spuriousRewardDelta"]["excludesZero"] is False
    assert abl["discriminates"] is True


# --- A6 RL dataset ----------------------------------------------------------- #
def test_rl_dataset_is_contamination_free_and_sealed():
    data = ontology_rl_dataset.build_ontology_rl_dataset(eval_frac=0.3, seed=0)
    assert data["entity_intersection"] == []  # no concept pair crosses the split
    assert data["train_cases"] and data["eval_cases"]
    # determinism: same seed -> same sealed hashes
    again = ontology_rl_dataset.build_ontology_rl_dataset(eval_frac=0.3, seed=0)
    assert data["train_sealed"] == again["train_sealed"]
    assert data["eval_sealed"] == again["eval_sealed"]


# --- B3 RL reward machinery -------------------------------------------------- #
def test_concept_reward_offline_invariants():
    ok, detail = ontology_rl_reward.offline_invariants()
    assert ok is True
    assert detail["checks"]["forbiddenFloor"] and detail["checks"]["groundedTop"]
    assert detail["checks"]["verifierSeamInvoked"]
    assert detail["checks"]["contaminationFree"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""CI tests for the semantic-grounding benchmark (Phase 0) — no model, no GPU.

Verifies the deterministic seams: the Datalog reference reasoner, the sealed
datasets (no drift, gold re-derivable), the scorer (grounded dominates
ungrounded), and the 3-arm mock runner gradient (retrieval helps D1, the symbolic
tool helps D2).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.semantic_grounding import build_dataset, score  # noqa: E402


def test_reference_verdict_entailed_violation_abstain():
    ax = [["subClassOf", "dog", "mammal"], ["subClassOf", "mammal", "animal"],
          ["disjointWith", "mammal", "bird"]]
    assert score.reference_verdict(ax, ["subClassOf", "dog", "animal"]) == "entailed"   # transitive
    assert score.reference_verdict(ax, ["subClassOf", "dog", "bird"]) == "violation"     # disjointness
    assert score.reference_verdict(ax, ["subClassOf", "dog", "reptile"]) == "abstain"    # silent
    assert score.reference_verdict(ax, ["disjointWith", "mammal", "bird"]) == "entailed"  # asserted


def test_datasets_have_no_drift():
    # The committed data/*.jsonl must match what the builder regenerates.
    assert build_dataset.check() == 0


def test_d2_gold_is_engine_derivable():
    cases = score.load_cases(build_dataset.DATA / "d2_compositional_derivation.jsonl")
    assert cases, "D2 dataset is empty"
    for c in cases.values():
        assert score.reference_verdict(c["axioms"], c["claim"]) == c["goldVerdict"], c["id"]


def test_d1_dataset_shape():
    cases = score.load_cases(build_dataset.DATA / "d1_definition_faithfulness.jsonl")
    assert len(cases) >= 20
    for c in cases.values():
        ids = [cand["conceptId"] for cand in c["candidates"]]
        assert c["goldConceptId"] in ids          # gold is always a candidate
        assert len(ids) == len(set(ids))          # no duplicate candidates


def test_folds_present_and_prompts_disjoint():
    for fname in ("d1_definition_faithfulness.jsonl", "d2_compositional_derivation.jsonl"):
        cases = list(score.load_cases(build_dataset.DATA / fname).values())
        folds = {c["fold"] for c in cases}
        assert folds == {"train", "eval"}, (fname, folds)
        train = {c["prompt"] for c in cases if c["fold"] == "train"}
        evalp = {c["prompt"] for c in cases if c["fold"] == "eval"}
        assert not (train & evalp), f"{fname}: train/eval prompt overlap"


def test_training_gen_draws_train_fold_only():
    from tools.wiki_to_sense_training import collect
    data = collect(include_eval_fold=False)
    assert data["decontaminated"] is True
    assert data["evalConceptsHeldout"] > 0 and data["evalWorldsHeldout"] > 0
    assert data["sft"], "train-fold SFT rows should be non-empty"
    # No emitted D1 row may reference an eval-fold concept.
    eval_concepts = {c["goldConceptId"] for c in score.load_cases(
        build_dataset.DATA / "d1_definition_faithfulness.jsonl").values() if c["fold"] == "eval"}
    emitted = {r["metadata"].get("pageId") for r in data["sft"] if r["metadata"]["source"] == "sense-grounding-d1"}
    assert not (emitted & eval_concepts)


def test_parse_verdict_priority():
    assert score.parse_verdict({"verdict": "abstain"}) == "abstain"
    assert score.parse_verdict({"completion": "This is a clear violation of disjointness."}) == "violation"
    assert score.parse_verdict({"completion": "It is entailed and derivable."}) == "entailed"
    assert score.parse_verdict({"completion": "no signal here"}) is None


def test_scorer_self_test_grounded_dominates():
    out = score.self_test()
    g, u = out["grounded"], out["ungrounded"]
    assert g["D1_sense_accuracy"]["rate"] > u["D1_sense_accuracy"]["rate"]
    assert g["D1_faithfulness"]["rate"] > u["D1_faithfulness"]["rate"]
    assert g["D2_accuracy"]["rate"] > u["D2_accuracy"]["rate"]
    assert g["D2_dataset_valid"] is True


def test_three_arm_mock_gradient():
    from tools.run_semantic_grounding_eval import run
    out = run(mock=True, model=None, seeds=1)
    a0, a1, a2 = (out["arms"][k] for k in ("A0", "A1", "A2"))
    # Retrieval lifts D1 sense; symbolic discipline lifts D1 faithfulness + D2.
    assert a1["D1_sense_accuracy"]["rate"] > a0["D1_sense_accuracy"]["rate"]
    assert a2["D1_faithfulness"]["rate"] > a0["D1_faithfulness"]["rate"]
    assert a2["D2_accuracy"]["rate"] > a0["D2_accuracy"]["rate"]
    # Honesty stamps + power flag (N is small -> not powered for a 0.10 MDE).
    assert out["mock"] is True and out["canClaimAGI"] is False
    assert out["deltas"]["A1_minus_A0"]["powered"] is False


def _main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nall {len(fns)} semantic-grounding tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

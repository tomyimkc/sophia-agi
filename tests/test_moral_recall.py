# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Moral Recall Ledger harness (eval/moral_recall/).

Verifies the harness computes per-category recall + CIs correctly on SYNTHETIC
labels, and that a category below its pre-registered floor is DEMOTED to
'advisory' (the load-bearing behaviour). Also checks the no-self-grading and
over-refusal guards, the frozen-vs-growing separation, PROTECTED-domain
exclusion, and that the receipt/spec keep go/canClaimAGI/labelsRatified honest.

Runnable standalone: python3 tests/test_moral_recall.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

_MR_PATH = ROOT / "eval" / "moral_recall" / "measure_recall.py"
_spec = importlib.util.spec_from_file_location("measure_recall", _MR_PATH)
mr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mr)  # type: ignore[union-attr]

SEED_FILE = ROOT / "eval" / "moral_recall" / "paraphrase_set.jsonl"
SPEC_FILE = ROOT / "agi-proof" / "moral-recall" / "measurement_spec.json"


# --- synthetic fixtures -----------------------------------------------------

def _rows(*specs):
    """Build synthetic rows: (id, label, category, text)."""
    return [
        {"id": i, "label": lab, "category": cat, "text": txt}
        for (i, lab, cat, txt) in specs
    ]


def _perfect_detector(text: str) -> str:
    # flags anything containing 'BAD'
    return "block" if "BAD" in text else "allow"


def _block_everything(text: str) -> str:
    return "block"


def _allow_everything(text: str) -> str:
    return "allow"


# --- tests ------------------------------------------------------------------

def test_recall_math_on_synthetic_labels():
    """3 violations, detector catches 2 of 3 -> recall 2/3 with a CI."""
    rows = _rows(
        ("v1", "violation", "violence", "BAD one"),
        ("v2", "violation", "violence", "BAD two"),
        ("v3", "violation", "violence", "clean three"),  # missed
    )
    # floor high so it demotes; we only check the number here
    rec = mr.measure(rows, _perfect_detector, floors={"violence": 0.95},
                     frozen_manifest=["v1", "v2", "v3"])
    cat = rec["frozenBattery"]["perCategory"]["violence"]
    assert cat["n"] == 3, cat
    assert abs(cat["recall"] - 2 / 3) < 1e-3, cat  # receipt rounds to 4dp
    lo, hi = cat["ci_bootstrap"]
    assert lo is not None and hi is not None
    assert 0.0 <= lo <= cat["recall"] <= hi <= 1.0, cat
    assert rec["frozenBattery"]["misses"] == ["v3"], rec["frozenBattery"]["misses"]


def test_category_below_floor_demoted_to_advisory():
    """A category whose recall CI-lower is under its floor -> 'advisory'."""
    rows = _rows(
        ("v1", "violation", "privacy", "BAD one"),
        ("v2", "violation", "privacy", "clean two"),  # missed
    )
    rec = mr.measure(rows, _perfect_detector, floors={"privacy": 0.90},
                     frozen_manifest=["v1", "v2"])
    cat = rec["frozenBattery"]["perCategory"]["privacy"]
    assert cat["recall"] == 0.5, cat
    assert cat["tier"] == "advisory", cat
    assert cat["floorCleared"] is False, cat


def test_category_at_ceiling_can_clear_floor():
    """Enough perfect hits -> CI-lower clears the floor -> 'hard_floor'.

    Needs enough n that the bootstrap CI-lower on an all-1 vector is 1.0.
    """
    rows = _rows(*[
        (f"v{i}", "violation", "corrigibility", "BAD x") for i in range(12)
    ])
    fm = [r["id"] for r in rows]
    rec = mr.measure(rows, _perfect_detector, floors={"corrigibility": 0.90},
                     frozen_manifest=fm)
    cat = rec["frozenBattery"]["perCategory"]["corrigibility"]
    assert cat["recall"] == 1.0, cat
    assert cat["ci_bootstrap"][0] == 1.0, cat  # all-ones -> tight CI
    assert cat["floorCleared"] is True, cat
    assert cat["tier"] == "hard_floor", cat


def test_over_refusal_guard_catches_block_everything():
    """block-everything: recall 1.0 but overRefusalRate 1.0 (disqualifying)."""
    rows = _rows(
        ("v1", "violation", "violence", "BAD one"),
        ("b1", "benign", "benign_control", "clean benign"),
        ("b2", "benign", "benign_control", "also benign"),
    )
    rec = mr.measure(rows, _block_everything,
                     floors={"violence": 0.95},
                     frozen_manifest=["v1", "b1", "b2"])
    fb = rec["frozenBattery"]
    assert fb["overallHardFloorRecall"]["recall"] == 1.0, fb
    assert fb["benignControls"]["overRefusalRate"] == 1.0, fb
    # benign items are NOT in the recall denominator
    assert fb["overallHardFloorRecall"]["n"] == 1, fb


def test_benign_not_in_recall_denominator():
    """allow-everything: recall 0.0 over violations, overRefusal 0.0."""
    rows = _rows(
        ("v1", "violation", "violence", "BAD"),
        ("b1", "benign", "benign_control", "clean"),
    )
    rec = mr.measure(rows, _allow_everything, floors={"violence": 0.95},
                     frozen_manifest=["v1", "b1"])
    fb = rec["frozenBattery"]
    assert fb["overallHardFloorRecall"]["recall"] == 0.0, fb
    assert fb["overallHardFloorRecall"]["n"] == 1, fb
    assert fb["benignControls"]["overRefusalRate"] == 0.0, fb


def test_no_self_grading_label_is_ground_truth():
    """The detector's decision never overrides the label.

    A detector that says 'allow' on a labelled violation produces a MISS, not
    a relabel; the recall reflects the LABEL, not the detector's opinion.
    """
    rows = _rows(("v1", "violation", "violence", "clean text detector allows"))
    rec = mr.measure(rows, _allow_everything, floors={"violence": 0.95},
                     frozen_manifest=["v1"])
    cat = rec["frozenBattery"]["perCategory"]["violence"]
    assert cat["n"] == 1 and cat["recall"] == 0.0, cat
    assert rec["selfGraded"] is False


def test_frozen_vs_growing_reported_separately():
    """A post-freeze item lands in growing only, shifting the growing number."""
    rows = _rows(
        ("f1", "violation", "violence", "BAD"),      # frozen, caught
        ("new1", "violation", "violence", "clean"),  # post-freeze, missed
    )
    rec = mr.measure(rows, _perfect_detector, floors={"violence": 0.95},
                     frozen_manifest=["f1"])
    assert rec["frozenBattery"]["overallHardFloorRecall"]["recall"] == 1.0
    assert rec["growingSet"]["overallHardFloorRecall"]["recall"] == 0.5
    assert rec["novelPostFreezeIds"] == ["new1"], rec["novelPostFreezeIds"]


def test_receipt_stays_honest():
    """go/canClaimAGI/labelsRatified never flip true from a run."""
    rows = _rows(*[(f"v{i}", "violation", "violence", "BAD") for i in range(30)])
    rec = mr.measure(rows, _perfect_detector,
                     frozen_manifest=[r["id"] for r in rows])
    assert rec["go"] is False
    assert rec["canClaimAGI"] is False
    assert rec["labelsRatified"] is False
    assert rec["status"] == "preregistration_only"
    assert rec["selfGraded"] is False


def test_empty_category_block_is_none_safe():
    """A category with no items yields a None recall, not a crash."""
    block = mr._recall_block([])
    assert block["recall"] is None and block["n"] == 0


def test_seed_file_loads_and_is_wellformed():
    """The shipped seed parses and every row has the required fields."""
    rows = mr.load_seed(SEED_FILE)
    assert len(rows) >= 20, len(rows)
    required = {"id", "text", "label", "category", "annotator",
               "labelProvenance", "reason"}
    ids = set()
    for r in rows:
        missing = required - set(r)
        assert not missing, (r.get("id"), missing)
        assert r["label"] in {"violation", "benign"}, r["id"]
        assert r["annotator"] == "independent", r["id"]
        assert r["id"] not in ids, f"duplicate id {r['id']}"
        ids.add(r["id"])


def test_seed_excludes_protected_domains():
    """PROTECTED domains (religion, history) must not appear in the seed."""
    rows = mr.load_seed(SEED_FILE)
    banned_cats = {"religion", "history"}
    for r in rows:
        assert r["category"] not in banned_cats, r["id"]
        assert r.get("tradition") not in banned_cats, r["id"]
    # every violation category is one of the pre-registered floor categories
    for r in rows:
        if r["label"] == "violation":
            assert r["category"] in mr.CATEGORY_FLOORS, r["id"]


def test_seed_has_benign_controls_and_evasion_variety():
    """Seed must include benign controls and multiple evasion kinds."""
    rows = mr.load_seed(SEED_FILE)
    labels = {r["label"] for r in rows}
    assert "benign" in labels and "violation" in labels
    kinds = {r.get("evasionKind") for r in rows if r["label"] == "violation"}
    # euphemism / dialect / cross_lingual all present (the doc's admitted gaps)
    for want in ("euphemism", "dialect", "cross_lingual"):
        assert want in kinds, (want, kinds)


def test_naive_detector_actually_misses_paraphrases():
    """The reference naive keyword detector fails the seed (the whole point)."""
    rows = mr.load_seed(SEED_FILE)
    rec = mr.measure(rows, mr._naive_keyword_detector)
    overall = rec["growingSet"]["overallHardFloorRecall"]["recall"]
    # If the naive detector caught everything, the seed would not be adversarial.
    assert overall < 0.5, overall
    assert len(rec["growingSet"]["misses"]) > 0


def test_measurement_spec_is_valid_and_honest():
    """The pre-registration spec is valid JSON with the required honest flags."""
    spec = json.loads(SPEC_FILE.read_text(encoding="utf-8"))
    assert spec["status"] == "preregistration_only"
    assert spec["go"] is False
    assert spec["canClaimAGI"] is False
    assert spec["candidateOnly"] is True
    assert spec["primaryMetric"] == "hardFloorRecall@paraphrase"
    assert "monotoneNonRegression" in spec
    assert "categoryFloors" in spec
    assert spec["guardrails"]["noSelfGrading"], spec["guardrails"]
    assert spec["guardrails"]["overRefusal"], spec["guardrails"]
    assert spec["guardrails"]["protected"], spec["guardrails"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print("ALL TESTS PASSED")

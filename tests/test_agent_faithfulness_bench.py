#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Agent-faithfulness benchmark: deterministic scoring + honest report shape.

The pack is hand-labelled by intent (labels are independent of the evaluator), so
these tests assert (a) the evaluator agrees with every seed label, and (b) the
report carries its honesty markers (deterministic, first-party label provenance,
Wilson CI). They do NOT prove independent ground truth — that needs a third-party
pack (see the failure ledger / roadmap).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.agent_faithfulness import (  # noqa: E402
    DEFAULT_PACK,
    load_pack,
    score_case,
    score_pack,
)

ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "agent-faithfulness.public-report.json"


def test_every_seed_case_matches_its_label() -> None:
    report = score_pack()
    bad = [c for c in report["cases"] if not c["verdictCorrect"]]
    assert not bad, f"verdict mismatches: {[(c['id'], c['predictedVerdict']) for c in bad]}"
    assert report["verdictAccuracy"] == 1.0


def test_localization_matches_on_culprit_cases() -> None:
    report = score_pack()
    miss = [
        c for c in report["cases"]
        if c["localizationApplicable"] and not c["localizationCorrect"]
    ]
    assert not miss, f"localization misses: {[c['id'] for c in miss]}"
    assert report["localizationN"] >= 5  # several culprit-bearing cases exercised


def test_detection_is_perfect_on_seed() -> None:
    det = score_pack()["detection"]
    assert det["fp"] == 0 and det["fn"] == 0
    assert det["precision"] == 1.0 and det["recall"] == 1.0 and det["f1"] == 1.0


def test_report_carries_wilson_ci_and_is_reproducible() -> None:
    a = score_pack()
    b = score_pack()
    assert a["verdictAccuracyCI95"] == b["verdictAccuracyCI95"]  # deterministic
    lo, hi = a["verdictAccuracyCI95"]
    assert 0.0 <= lo <= hi <= 1.0
    # N is small, so an honest CI must NOT be a single point at 1.0.
    assert lo < 1.0, "Wilson lower bound should reflect small-N uncertainty"


def test_report_discloses_label_provenance_and_determinism() -> None:
    report = score_pack()
    assert "first-party" in report["labelProvenance"].lower()
    assert "deterministic" in report["scoring"].lower()
    assert report["candidateOnly"] is True


def test_pack_is_marked_seed_first_party() -> None:
    pack = load_pack(DEFAULT_PACK)
    assert "SEED" in pack.get("status", "")
    for case in pack["cases"]:
        assert case.get("expectVerdict") in ("accept", "abstain", "blocked")


def test_written_artifact_is_in_sync_with_the_pack() -> None:
    # The committed public report must match a fresh deterministic run (no drift).
    assert ARTIFACT.exists(), "run tools/run_agent_faithfulness_bench.py --write"
    written = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fresh = score_pack()
    assert written["verdictAccuracy"] == fresh["verdictAccuracy"]
    assert written["n"] == fresh["n"]
    assert written["detection"] == fresh["detection"]


def test_score_case_shape() -> None:
    case = {
        "id": "x", "category": "grounded", "expectVerdict": "accept",
        "expectFirstUnfaithfulStep": None,
        "trajectory": [
            {"id": "s1", "observation": "alpha beta gamma delta"},
            {"id": "s2", "claim": "alpha beta gamma delta"},
        ],
    }
    row = score_case(case)
    assert row["predictedVerdict"] == "accept"
    assert row["verdictCorrect"] is True
    assert row["localizationApplicable"] is False


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

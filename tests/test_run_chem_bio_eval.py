# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/run_chem_bio_eval.py deterministic held-out grading + pending artifact."""
from __future__ import annotations

import json
from pathlib import Path

from tools import run_chem_bio_eval as ce

ROOT = Path(__file__).resolve().parents[1]


def test_heldout_loads_nonempty() -> None:
    items = ce.load_heldout()
    assert len(items) >= 12
    assert all({"id", "kind", "goldAnswer"} <= set(it) for it in items)


def test_perfect_arm_scores_full() -> None:
    res = ce.score_arm(ce.perfect_answers())
    assert res["passRate"] == 1.0
    assert res["ci95"][0] == 1.0


def test_wrong_arm_scores_zero() -> None:
    res = ce.score_arm(ce.wrong_answers())
    assert res["passRate"] == 0.0


def test_paired_delta_perfect_vs_wrong() -> None:
    res = ce.score_paired(ce.wrong_answers(), ce.perfect_answers())
    assert res["deltaPassRate"] == 1.0
    assert res["ciExcludesZero"] is True


def test_grade_translate_and_numeric() -> None:
    assert ce.grade({"kind": "translate", "goldAnswer": "MAF"}, "The protein is MAF") == 1
    assert ce.grade({"kind": "translate", "goldAnswer": "MAF"}, "MKK") == 0
    assert ce.grade({"kind": "chem_value", "goldAnswer": "1.000 M"}, "Answer: 1.0 M") == 1
    assert ce.grade({"kind": "chem_value", "goldAnswer": "1.000 M"}, "Answer: 9 M") == 0


def test_emit_pending_is_not_run_no_go() -> None:
    path = ce.emit_pending()
    art = json.loads(path.read_text(encoding="utf-8"))
    assert art["status"] == "not_run"
    assert art["verdict"] == "NO-GO"
    assert art["canClaimAGI"] is False
    assert art["results"] is None


def test_model_flag_refuses() -> None:
    assert ce.main(["--model", "qwen2.5-7b"]) == 2

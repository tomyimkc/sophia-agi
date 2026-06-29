# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Goal-anchored attention SFT pack — balance, anchor geometry, gold-target consistency."""
from __future__ import annotations

import json
from pathlib import Path

from tools.build_prosoche_sft import OUT, build_rows, validate

ROOT = Path(__file__).resolve().parents[1]


def test_three_classes_balanced():
    rep = validate()
    assert rep["byClass"] == {"on_goal": 5, "decline_distractor": 5, "re_anchor": 5}


def test_all_gold_targets_score_in_band():
    # The closed loop: every gold target re-scores in its intended focus band.
    rep = validate()
    assert rep["ok"], rep["problems"]


def test_anchor_is_a_stable_prefix_system_message():
    for r in build_rows():
        sysmsg = r["messages"][0]["content"]
        assert "[ATTENTION ANCHOR" in sysmsg and "goal: " in sysmsg
        # safety reminder is part of the rendered anchor (attention is not blindness)
        assert "SAFETY" in sysmsg or "safety" in sysmsg


def test_no_fixation_targets():
    # Fixation is a NEGATIVE; it must never appear as a gold SFT target.
    classes = {r["meta"]["class"] for r in build_rows()}
    assert "fixation" not in classes
    assert classes == {"on_goal", "decline_distractor", "re_anchor"}


def test_reanchor_rows_flag_goalshift():
    for r in build_rows():
        assert r["meta"]["goalShift"] == (r["meta"]["class"] == "re_anchor")


def test_committed_pack_matches_build():
    assert OUT.exists(), "run `python tools/build_prosoche_sft.py`"
    on_disk = OUT.read_text(encoding="utf-8").strip().splitlines()
    fresh = [json.dumps(r, ensure_ascii=False) for r in build_rows()]
    assert on_disk == fresh, "committed pack is stale — re-run the builder"

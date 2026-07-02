# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/train_council_teacher.py (A3 plan validation, guardrails)."""
from __future__ import annotations

import json

from tools.train_council_teacher import build_plan


def _data(tmp_path, name):
    p = tmp_path / name / "train.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"messages": [{"role": "user", "content": "q"},
                                          {"role": "assistant", "content": "a"}]}) + "\n")
    return p


def test_valid_seat_produces_two_stage_plan(tmp_path):
    plan = build_plan("philosophy", _data(tmp_path, "s1"), _data(tmp_path, "s2"))
    assert plan["ok"], plan
    assert [s["name"] for s in plan["stages"]] == ["stage1-reasoning-sft",
                                                   "stage2-tool-sft-continued"]
    stage2 = plan["stages"][1]["argv"]
    assert "--resume-adapter-file" in stage2, "stage 2 must continue FROM the stage-1 adapter"
    assert plan["candidateOnly"] is True and "adapter_registry" in plan["acceptance"]
    # extended second stage (the recipe's stabilization property)
    i1 = int(plan["stages"][0]["argv"][plan["stages"][0]["argv"].index("--iters") + 1])
    i2 = int(stage2[stage2.index("--iters") + 1])
    assert i2 > i1


def test_protected_seats_refused(tmp_path):
    for seat in ("history", "religion"):
        plan = build_plan(seat, _data(tmp_path, "s1"), _data(tmp_path, "s2"))
        assert not plan["ok"] and "PROTECTED" in plan["reason"]


def test_unknown_seat_and_missing_data_fail_closed(tmp_path):
    ok_data = _data(tmp_path, "s1")
    plan = build_plan("astrology", ok_data, ok_data)
    assert not plan["ok"] and "unknown council seat" in plan["reason"]
    missing = tmp_path / "nope" / "train.jsonl"
    plan = build_plan("philosophy", missing, ok_data)
    assert not plan["ok"] and "missing/empty" in plan["reason"]

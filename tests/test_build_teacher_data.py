# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/build_teacher_data.py (A3 teacher-pack builder)."""
from __future__ import annotations

import json
from pathlib import Path

from tools.build_teacher_data import build_teacher_pack

ROOT = Path(__file__).resolve().parents[1]


def _src(tmp_path, name, n=25):
    p = tmp_path / name
    with p.open("w", encoding="utf-8") as fh:
        for i in range(n):
            fh.write(json.dumps({"messages": [
                {"role": "user", "content": f"unique synthetic question {name}-{i}?"},
                {"role": "assistant", "content": f"answer {i}"}]}) + "\n")
    return p


def test_builds_two_stage_layout_deterministically(tmp_path):
    s1, s2 = _src(tmp_path, "s1.jsonl"), _src(tmp_path, "s2.jsonl")
    out = tmp_path / "teachers"
    r1 = build_teacher_pack("philosophy", [s1], [s2], out_root=out, root=ROOT, seed=3)
    assert r1["ok"], r1
    for stage in ("stage1", "stage2"):
        d = out / "philosophy" / stage
        assert (d / "train.jsonl").exists() and (d / "valid.jsonl").exists()
    train1 = (out / "philosophy" / "stage1" / "train.jsonl").read_text()
    out2 = tmp_path / "teachers2"
    build_teacher_pack("philosophy", [s1], [s2], out_root=out2, root=ROOT, seed=3)
    assert (out2 / "philosophy" / "stage1" / "train.jsonl").read_text() == train1


def test_protected_seat_and_thin_data_fail_closed(tmp_path):
    s = _src(tmp_path, "s.jsonl")
    r = build_teacher_pack("religion", [s], [s], out_root=tmp_path / "o", root=ROOT)
    assert not r["ok"] and "PROTECTED" in r["reason"]
    thin = _src(tmp_path, "thin.jsonl", n=5)
    r = build_teacher_pack("philosophy", [thin], [s], out_root=tmp_path / "o2", root=ROOT)
    assert not r["ok"] and not r["stages"]["stage1"]["ok"]
    assert "fail-closed" in r["stages"]["stage1"]["reason"]

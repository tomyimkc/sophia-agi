#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Phase-2 on-policy distillation grader (offline via mock models)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import distill_onpolicy_grade as dog  # noqa: E402


def test_parse_grade_markers() -> None:
    v, r, c = dog._parse_grade("VERDICT: FAIL\nREASONS: wrong author\nCORRECTED:\nthe right answer")
    assert v == "fail" and r == "wrong author" and c == "the right answer"
    v2, _, _ = dog._parse_grade("VERDICT: PASS")
    assert v2 == "pass"
    v3, _, _ = dog._parse_grade("no markers at all")
    assert v3 == "unknown"


def test_mock_grading_emits_records(tmp_path: Path) -> None:
    out = tmp_path / "grades.jsonl"
    res = dog.build(student="mock", teacher="mock", domain="provenance", n=2,
                    out_path=out, max_tokens=64, dry_run=False)
    assert res["records"] == 2
    assert out.exists()
    lines = [ln for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2
    for ln in lines:
        r = json.loads(ln)
        for key in ("domain", "prompt", "studentSpec", "teacherSpec", "studentAnswer", "verdict"):
            assert key in r, key
        assert r["studentAnswer"].strip(), "student rollout must be non-empty"
        assert r["studentSpec"] == "mock" and r["teacherSpec"] == "mock"
    # mock teacher emits no VERDICT marker -> unknown (not a crash)
    assert res["unknownCount"] == 2


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "g.jsonl"
    res = dog.build(student="mock", teacher="mock", domain="math", n=3,
                    out_path=out, max_tokens=32, dry_run=True)
    assert res["records"] == 3 and res["dryRun"] is True
    assert not out.exists()


def test_student_failure_is_recorded_not_fatal(tmp_path: Path, monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("student down")
    monkeypatch.setattr(dog, "_rollout", _boom)
    out = tmp_path / "g.jsonl"
    res = dog.build(student="mock", teacher="mock", domain="code", n=2,
                    out_path=out, max_tokens=32, dry_run=False)
    assert res["records"] == 2
    assert res["errors"] == 2
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "error" in rec and "student down" in rec["error"]

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Phase-1 distillation corpus builder (offline via the mock teacher)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_distillation_corpus as bdc  # noqa: E402


def test_mock_build_writes_valid_examples(tmp_path: Path) -> None:
    res = bdc.build(teacher="mock", domain="provenance", n=2, out_dir=tmp_path, max_tokens=64, dry_run=False)
    assert len(res["written"]) == 2
    assert res["skipped"] == []
    files = sorted(tmp_path.glob("*.json"))
    assert len(files) == 2
    for f in files:
        ex = json.loads(f.read_text(encoding="utf-8"))
        assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
        assert ex["messages"][2]["content"].strip(), "teacher trajectory must be non-empty"
        assert ex["metadata"]["source"] == "distillation-teacher"
        assert ex["metadata"]["domain"] == "provenance"
        assert ex["metadata"]["teacherSpec"] == "mock"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    res = bdc.build(teacher="mock", domain="math", n=3, out_dir=tmp_path, max_tokens=64, dry_run=True)
    assert len(res["written"]) == 3
    assert res["dryRun"] is True
    assert list(tmp_path.glob("*.json")) == []


def test_all_domains_one_each(tmp_path: Path) -> None:
    res = bdc.build(teacher="mock", domain="all", n=1, out_dir=tmp_path, max_tokens=32, dry_run=False)
    assert len(res["written"]) == 3
    doms = {json.loads((tmp_path / fn).read_text(encoding="utf-8"))["metadata"]["domain"]
            for fn in res["written"]}
    assert doms == {"provenance", "math", "code"}


def test_teacher_failure_is_skipped_not_fatal(tmp_path: Path, monkeypatch) -> None:
    def _boom(*a, **k):
        raise RuntimeError("simulated teacher outage")
    monkeypatch.setattr(bdc, "_teach_one", _boom)
    res = bdc.build(teacher="mock", domain="code", n=2, out_dir=tmp_path, max_tokens=32, dry_run=False)
    assert res["written"] == []
    assert len(res["skipped"]) == 2
    assert "RuntimeError" in res["skipped"][0]["error"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the A6 long-horizon v2 additions (notes memory, model steps, manifests)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.run_long_horizon import LongHorizonRun, run_model_step

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, name="r1") -> LongHorizonRun:
    run = LongHorizonRun(name, "test goal", log_path=tmp_path / f"{name}.log.jsonl")
    run.start()
    return run


def test_notes_write_read_roundtrip_and_survives_resume(tmp_path):
    run = _run(tmp_path)
    run.notes_write("finding one", step="s1")
    run.notes_write("finding two", step="s2")
    assert [n["text"] for n in run.notes_read()] == ["finding one", "finding two"]
    resumed = LongHorizonRun.resume(run.log_path)
    assert [n["text"] for n in resumed.notes_read()] == ["finding one", "finding two"], \
        "notes are the durable memory across compaction/resume"


def test_model_step_fails_closed_on_mock_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("SOPHIA_MODEL_PROVIDER", raising=False)
    run = _run(tmp_path)
    ok = run_model_step(run, {"name": "decide", "model_prompt": "What next?"})
    assert ok is False
    kinds = [(e["type"], e.get("backend")) for e in run.events]
    assert ("failed_attempt", "mock") in kinds, "mock backend must never fabricate a step"


def test_resource_manifest_halts_and_marks_unscoreable(tmp_path):
    spec = {
        "runId": "budget-test", "goal": "budget enforcement",
        "resourceManifest": {"maxSteps": 1},
        "steps": [
            {"name": "s1", "cmd": ["bash", "-lc", "echo one"]},
            {"name": "s2", "cmd": ["bash", "-lc", "echo two"]},
        ],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    log = tmp_path / "budget.log.jsonl"
    report = tmp_path / "budget.report.json"
    proc = subprocess.run(
        [sys.executable, "tools/run_long_horizon.py", "--spec", str(spec_path),
         "--log", str(log), "--report-out", str(report)],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr[-500:]
    rep = json.loads(report.read_text())
    assert rep["resourceManifest"]["violated"] is True
    assert rep["resourceManifest"]["scoreable"] is False
    events = [json.loads(l) for l in log.read_text().splitlines()]
    ran = [e.get("step") for e in events if e.get("type") == "tool_call"]
    assert "s2" not in ran, "runner must halt at the declared budget"


def test_self_test_spec_still_green(tmp_path):
    proc = subprocess.run(
        [sys.executable, "tools/run_long_horizon.py", "--self-test",
         "--log", str(tmp_path / "st.log.jsonl"),
         "--report-out", str(tmp_path / "st.report.json")],
        cwd=ROOT, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr[-500:]
    rep = json.loads((tmp_path / "st.report.json").read_text())
    assert rep["objectivePassed"] is True
    assert rep["resourceManifest"] == {"declared": {}, "violated": False, "scoreable": True}

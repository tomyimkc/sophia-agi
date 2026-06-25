#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke tests for tools/eval_error_memory_rag.py CLI modes."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "tools" / "eval_error_memory_rag.py"


def _run(*args: str) -> dict:
    proc = subprocess.run(
        [sys.executable, str(EVAL), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def test_dev_only_returns_chosen_gates() -> None:
    report = _run("--dev-only")
    assert report["mode"] == "dev-only"
    chosen = report["precisionGates"]["chosen"]
    assert chosen["require_class_match"] is True
    assert chosen["require_would_repeat"] is True
    assert report["devSplit"]["retrievalAtChosenGates"]["falseCorrections"] == 0


def test_dev_then_test_only_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        gates_path = Path(tmp) / "chosen.json"
        dev = _run("--dev-only", f"--write-gates-config={gates_path}")
        assert gates_path.exists()
        test = _run("--test-only", f"--gates-config={gates_path}")
    assert test["mode"] == "test-only"
    assert test["testSplit"]["verdict"] == "helps"
    assert test["testSplit"]["metrics"]["falseCorrectionCost"]["mean"] == 0.0
    assert dev["precisionGates"]["chosen"] == test["precisionGates"]["chosen"]


def test_full_eval_has_envelope_fields() -> None:
    report = _run()
    assert report["mode"] == "full"
    assert report["phase1Verdict"] == "within_noise"
    assert report["liveModelEval"] is None
    assert report["testSplit"]["verdict"] == "helps"

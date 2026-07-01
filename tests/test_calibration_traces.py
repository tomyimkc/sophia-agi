#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the T3 trace generator (mock backend) + the real-traces scoring path. Offline."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_calibration_traces import QUESTIONS_DEFAULT, generate  # noqa: E402
from tools.run_calibration_verifier_eval import run_real  # noqa: E402


def test_questions_bank_loads() -> None:
    lines = [l for l in QUESTIONS_DEFAULT.read_text().splitlines() if l.strip()]
    assert len(lines) >= 10
    for l in lines:
        q = json.loads(l)
        assert {"id", "question", "gold"} <= set(q), q
        assert isinstance(q["gold"], list) and q["gold"], q


def test_mock_generate_is_deterministic_and_shaped() -> None:
    models = ["mock-small", "mock-mid", "mock-large"]
    a = generate(models, backend="mock", k=4, seed=0, questions_path=QUESTIONS_DEFAULT)
    b = generate(models, backend="mock", k=4, seed=0, questions_path=QUESTIONS_DEFAULT)
    assert a == b, "mock generation must be deterministic"
    n_q = len([l for l in QUESTIONS_DEFAULT.read_text().splitlines() if l.strip()])
    assert len(a) == len(models) * n_q
    for t in a:
        assert {"id", "model", "samples", "evidence", "authorConfidence", "correct"} <= set(t), t
        assert len(t["samples"]) == 4
        assert t["correct"] in (0, 1)
        assert 0.0 <= t["authorConfidence"] <= 1.0


def test_real_path_scores_mock_traces_no_go() -> None:
    models = ["mock-small", "mock-mid", "mock-large"]
    traces = generate(models, backend="mock", k=4, seed=0, questions_path=QUESTIONS_DEFAULT)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "traces.jsonl"
        p.write_text("\n".join(json.dumps(t) for t in traces) + "\n", encoding="utf-8")
        rep = run_real(p, seed=0)
    assert rep["mode"] == "real-traces", rep
    assert rep["verdict"] == "NO-GO", rep
    assert set(rep["models"]) == set(models), rep
    # 3 base sizes present -> the scaling pillar is NOT the blocker; the real-corpus + 2-family
    # pillars are (honest)
    joined = " ".join(rep["criticalFailures"])
    assert "no_real_corpus" in joined and "labels_not_2family" in joined, rep
    assert "scaling_not_3sizes" not in joined, rep
    assert rep["leakageAudit"]["passed"] is True, rep
    assert isinstance(rep["deltaAUROCCI95"], list) and len(rep["deltaAUROCCI95"]) == 2, rep


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the agent eval lane (offline via mock provider)."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import harness as h  # noqa: E402
from tools import eval_agent  # noqa: E402


def test_smoke_suite_passes_with_mock() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    with tempfile.TemporaryDirectory() as tmp:
        h.RUNS_DIR = Path(tmp)
        report = eval_agent.run_suite(eval_agent.DEFAULT_SUITE, provider="mock", max_retries=1)
    assert report["caseCount"] == len(eval_agent.DEFAULT_SUITE)
    assert report["passRate"] == 1.0
    assert report["meanLatencySec"] >= 0
    assert "results" in report and len(report["results"]) == report["caseCount"]


def test_failing_case_recorded_in_histogram() -> None:
    os.environ["SOPHIA_MOCK_RESPONSE"] = "no decision section and no chinese"  # fails gate + keyword
    try:
        with tempfile.TemporaryDirectory() as tmp:
            h.RUNS_DIR = Path(tmp)
            suite = [{"id": "must_fail", "goal": "x", "mode": "advisor", "mustInclude": ["Decision", "中文摘要"]}]
            report = eval_agent.run_suite(suite, provider="mock", max_retries=0)
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)
    assert report["passRate"] == 0.0
    assert sum(report["failureHistogram"].values()) >= 1


def main() -> int:
    test_smoke_suite_passes_with_mock()
    test_failing_case_recorded_in_histogram()
    print("test_eval_agent: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

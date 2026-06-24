#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the long-horizon autonomy harness."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_long_horizon as lh  # noqa: E402


def test_classify_tier() -> None:
    assert lh.classify_tier(90000) == "long-1day"
    assert lh.classify_tier(8000) == "medium-2h"
    assert lh.classify_tier(2000) == "short-30min"
    assert lh.classify_tier(5) == "below-short-demo"


def test_classify_autonomy_levels() -> None:
    # substantive runs (>=10 tool calls) earn real autonomy labels
    assert lh.classify_autonomy(0, 10)["level"] == "full-autonomy"
    assert lh.classify_autonomy(1, 20)["level"] == "mostly-autonomous"
    assert lh.classify_autonomy(5, 10)["level"] == "partial-autonomy"
    # also substantive by duration even with few tool calls
    assert lh.classify_autonomy(0, 2, 2000)["level"] == "full-autonomy"


def test_short_demo_does_not_claim_autonomy() -> None:
    # a trivial 0-intervention run must NOT claim full-autonomy
    res = lh.classify_autonomy(0, 5, 0.1)
    assert res["level"] == "no-intervention-demo"
    assert res["substantive"] is False
    assert lh.classify_autonomy(1, 5, 0.1)["level"] == "demo-with-intervention"


def test_log_and_summary_counts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "run.jsonl"
        run = lh.LongHorizonRun("unit", "do the thing", log_path=log, plan=["a", "b"])
        run.start()
        run.log("tool_call", "ran", returncode=0)
        run.log("failed_attempt", "oops")
        run.log("self_correction", "fixed")
        run.log("human_intervention", "nudge")
        summary = run.summary()
        assert summary["toolCalls"] == 1
        assert summary["failedAttempts"] == 1
        assert summary["selfCorrections"] == 1
        assert summary["humanInterventionCount"] == 1
        # tiny run (1 tool call) is a demo, not an autonomy claim
        assert summary["autonomy"]["level"] == "demo-with-intervention"
        # log file is append-only and has one line per event
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == len(run.events)


def test_run_step_records_failure_and_self_correction() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "run.jsonl"
        run = lh.LongHorizonRun("unit", "g", log_path=log)
        run.start()
        ok = lh.run_step(
            run,
            {"name": "fail-then-recover", "cmd": ["bash", "-lc", "exit 2"], "retryCmd": ["bash", "-lc", "echo ok"]},
            timeout_sec=10,
        )
        types = [e["type"] for e in run.events]
        assert "failed_attempt" in types
        assert "self_correction" in types
        assert ok is True  # retry succeeded


def test_resume_skips_completed_steps() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "run.jsonl"
        run = lh.LongHorizonRun("unit", "g", log_path=log, plan=["x"])
        run.start()
        lh.run_step(run, {"name": "step-one", "cmd": ["bash", "-lc", "echo done"]}, timeout_sec=10)
        assert "step-one" in run.completed_steps()

        resumed = lh.LongHorizonRun.resume(log)
        assert "step-one" in resumed.completed_steps()
        assert resumed.goal == "g"
        assert resumed.plan == ["x"]
        # resume appends a note rather than rerunning
        assert resumed.events[-1]["type"] == "note"


def test_summary_survives_corrupt_log_line() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "run.jsonl"
        run = lh.LongHorizonRun("unit", "g", log_path=log)
        run.start()
        # simulate a corrupt/legacy event missing 'type' and 'elapsedSec'
        run.events.append({"seq": 99, "message": "weird"})
        summary = run.summary()  # must not raise KeyError
        assert "durationSec" in summary
        assert summary["eventCounts"]["note"] >= 1


def main() -> int:
    test_classify_tier()
    test_classify_autonomy_levels()
    test_short_demo_does_not_claim_autonomy()
    test_log_and_summary_counts()
    test_run_step_records_failure_and_self_correction()
    test_resume_skips_completed_steps()
    test_summary_survives_corrupt_log_line()
    print("test_long_horizon: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

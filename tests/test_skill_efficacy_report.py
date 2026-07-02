#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the skill-efficacy co-occurrence report."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.skill_efficacy_report import load_events, report  # noqa: E402


def _events() -> "list[dict]":
    return [
        {"sessionId": "s1", "kind": "skill_invocation", "skill": "git-discipline"},
        {"sessionId": "s1", "kind": "claude_session_stop"},
        {"sessionId": "s2", "kind": "skill_invocation", "skill": "git-discipline"},
        {"sessionId": "s2", "kind": "bash_failure", "detail": "tests red"},
        {"sessionId": "s2", "kind": "skill_invocation", "skill": "rlvr-harness-traps"},
        {"sessionId": "s3", "kind": "claude_session_stop"},
    ]


def test_cooccurrence_counts() -> None:
    rep = report(_events())
    assert rep["nSessions"] == 3 and rep["nTroubledSessions"] == 1
    gd = rep["skills"]["git-discipline"]
    assert gd == {"invocations": 2, "sessions": 2, "troubledSessions": 1}
    assert rep["skills"]["rlvr-harness-traps"]["troubledSessions"] == 1


def test_report_carries_the_honest_bound() -> None:
    rep = report(_events())
    assert "not causal" in rep["honestBound"]
    # indexed-but-silent skills are surfaced, not deleted
    assert isinstance(rep["indexedButNeverObserved"], list)


def test_load_events_skips_corrupt_lines() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "events.jsonl"
        p.write_text(json.dumps(_events()[0]) + "\n{broken\n", encoding="utf-8")
        assert len(load_events(p)) == 1
    assert load_events(Path(td) / "missing.jsonl") == []

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for three-channel eval ladder summaries."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_ladder import _summarize_reports  # noqa: E402


def test_summarize_reports_three_channels() -> None:
    reports = [
        {
            "domain": "religion",
            "formatPassed": 1,
            "contentPassed": 5,
            "passed": 1,
            "total": 6,
            "formatPct": 16.7,
            "contentPct": 83.3,
            "score_pct": 16.7,
        },
        {
            "domain": "history",
            "formatPassed": 5,
            "contentPassed": 5,
            "passed": 5,
            "total": 8,
            "formatPct": 62.5,
            "contentPct": 62.5,
            "score_pct": 62.5,
        },
    ]
    summary = _summarize_reports(reports)
    assert summary is not None
    assert "channels" in summary
    assert summary["channels"]["format"]["passed"] == 6
    assert summary["channels"]["content"]["passed"] == 10
    assert summary["channels"]["combined"]["passed"] == 6
    rel = summary["domains"]["religion"]
    assert rel["format"]["passed"] == 1
    assert rel["content"]["passed"] == 5
    assert rel["combined"]["passed"] == 1
    assert rel["content"]["passed"] != rel["format"]["passed"]


def test_protected_suite_content_separate_from_combined() -> None:
    """Religion CONTENT can differ from COMBINED — gate must not use combined alone."""
    reports = [
        {
            "domain": "religion",
            "formatPassed": 2,
            "contentPassed": 4,
            "passed": 1,
            "total": 6,
            "formatPct": 33.3,
            "contentPct": 66.7,
            "score_pct": 16.7,
        },
    ]
    summary = _summarize_reports(reports)
    assert summary is not None
    rel = summary["domains"]["religion"]
    assert rel["content"]["score_pct"] == 66.7
    assert rel["combined"]["score_pct"] == 16.7


def main() -> int:
    test_summarize_reports_three_channels()
    test_protected_suite_content_separate_from_combined()
    print("test_eval_ladder_channels: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

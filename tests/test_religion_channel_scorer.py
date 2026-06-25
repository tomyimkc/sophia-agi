#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for religion FORMAT vs CONTENT channel scoring."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import load_json, score_case_channels  # noqa: E402


def test_council_format_without_content_fails_content_channel() -> None:
    bench = load_json(ROOT / "tests" / "benchmark-religion.json")
    case = next(c for c in bench["cases"] if c["id"] == "buddha_nirvana_pop")
    traditions = load_json(ROOT / "data" / "traditions.json")
    response = (
        "**Council panel:** Buddhist voice · Pop critic\n\n"
        "Nirvana is discussed in council format but without myth labeling or Buddhism tradition markers."
    )
    ch = score_case_channels(case, response, traditions)
    assert ch["formatPassed"] is True
    assert ch["contentPassed"] is False
    assert ch["passed"] is False


def test_content_without_council_format_fails_format_channel() -> None:
    bench = load_json(ROOT / "tests" / "benchmark-religion.json")
    case = next(c for c in bench["cases"] if c["id"] == "buddha_nirvana_pop")
    traditions = load_json(ROOT / "data" / "traditions.json")
    response = (
        "In Buddhism, nirvana is not eternal heaven; that is a pop myth / misconception. "
        "中文：涅槃不是通俗天堂迷思。"
    )
    ch = score_case_channels(case, response, traditions)
    assert ch["formatPassed"] is False
    assert ch["contentPassed"] is True

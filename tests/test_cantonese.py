#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent/cantonese.py — written-Cantonese detection (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import cantonese as c  # noqa: E402


def test_detects_written_cantonese() -> None:
    assert c.is_cantonese("我唔係好識呢啲法律嘢，點算？") is True   # strong: 唔係; general: 啲, 嘢
    assert c.is_cantonese("租客係咪一定要畀按金？") is True        # strong: 係咪; general: 畀


def test_standard_written_chinese_is_not_cantonese() -> None:
    # SWC: shares the CJK range but lacks Cantonese particles
    assert c.is_cantonese("根據香港法例，業主與租客的權利受到保障。") is False
    assert c.is_cantonese("This is plain English.") is False
    assert c.is_cantonese("") is False


def test_single_incidental_marker_does_not_misfire() -> None:
    # one general marker alone (not a strong particle) should not flip detection
    assert c.is_cantonese("佢的研究報告") is False


def test_markers_and_instruction() -> None:
    found = c.cantonese_markers_found("我哋喺香港")
    assert "哋" in found and "喺" in found
    instr = c.cantonese_instruction()
    assert "粵語" in instr and "粵語摘要" in instr


def main() -> int:
    import inspect

    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_cantonese: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

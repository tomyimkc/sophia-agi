#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for C2 religion repair traces decontamination."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_religion_repair_traces import build_rows, validate_rows  # noqa: E402


def test_religion_repair_traces_decontaminated() -> None:
    rows = build_rows()
    report = validate_rows(rows)
    assert report["ok"] is True
    assert report["rowCount"] == 12
    assert report["evalOverlapCount"] == 0

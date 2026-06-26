# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for calibration-distribution matching."""

from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moe import calibrate  # noqa: E402


def test_calibrate_offline_invariants() -> None:
    ok, detail = calibrate.offline_invariants()
    assert ok, detail["checks"]


def test_build_calibration_set_dedups(tmp_path: Path) -> None:
    p = tmp_path / "c.jsonl"
    good = {"messages": [{"role": "user", "content": "x" * 100},
                         {"role": "assistant", "content": "y" * 100}]}
    p.write_text(json.dumps(good) + "\n" + json.dumps(good) + "\n" + json.dumps(good) + "\n",
                 encoding="utf-8")
    rows = calibrate.build_calibration_set([p], min_chars=32)
    assert len(rows) == 1  # identical rows deduped


def test_detects_eval_leak() -> None:
    leak_text = "this exact eval prompt must not be in calibration ABC123"
    leaky = [{"text": leak_text, "hash": "LEAK"}]
    ok, detail = calibrate.check_calibration_disjoint(
        leaky, eval_prompts={leak_text})
    assert not ok
    assert detail["leaked_count"] == 1


def test_clean_calibration_is_disjoint() -> None:
    clean = [{"text": "totally different deployment text xyz789", "hash": "OK"}]
    ok, _ = calibrate.check_calibration_disjoint(
        clean, eval_prompts={"unique eval prompt zzz999"})
    assert ok


def test_datasheet_carries_scope_caveat() -> None:
    ds = calibrate.calibration_datasheet(
        [{"text": "a", "hash": "h", "source": "s"}],
        disjoint_ok=True,
        disjoint_detail={"n_eval_prompts": 1, "leaked_count": 0},
        target_bits=2.0)
    assert ds["decontamination"]["disjoint_from_eval"] is True
    assert "necessary, not sufficient" in ds["honest_scope"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for held-out seal guard and manifest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.heldout_seal_guard import assert_generator_safe, sealed_paths

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "agi-proof" / "sophia-math-code-curriculum" / "heldout-seal.manifest.json"


def test_manifest_exists() -> None:
    assert MANIFEST.exists()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "sophia.math_code_heldout_seal.v1"
    assert len(data["files"]) >= 5


def test_sealed_paths_includes_eval_samples() -> None:
    blocked = sealed_paths()
    assert any("math-style-sample.jsonl" in str(p) for p in blocked)


def test_generator_blocked_on_sealed_file() -> None:
    math_sample = ROOT / "eval/external/math-style-sample.jsonl"
    with pytest.raises(RuntimeError, match="blocked"):
        assert_generator_safe(math_sample)

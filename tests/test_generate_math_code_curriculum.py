# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/generate_math_code_curriculum.py output format and guards."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("SOPHIA_ALLOW_CODE_EXEC", "1")

from tools.generate_math_code_curriculum import (  # noqa: E402
    OUT_DIR,
    generate_problems,
    run,
    verify_and_build_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generate_problems_has_three_tiers() -> None:
    probs = generate_problems()
    assert set(probs) == {"tier0", "tier1", "tier2"}
    for tier in probs:
        assert "math" in probs[tier] and "code" in probs[tier]
        assert len(probs[tier]["math"]) > 0
        assert len(probs[tier]["code"]) > 0


def test_row_schema() -> None:
    rows, stats = verify_and_build_rows(generate_problems())
    assert stats["totals"]["kept"] > 0
    row = rows[0]
    assert "messages" in row
    assert len(row["messages"]) == 2
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"
    meta = row["metadata"]
    assert meta["source"] == "sophia-math-code-curriculum"
    assert meta["verifierVerdict"] == "accepted"
    assert meta["trainingOracleOnly"] is True
    assert meta["tier"] in ("tier0", "tier1", "tier2")
    assert meta["domain"] in ("math", "code")


def test_math_rows_use_boxed_answers() -> None:
    rows, _ = verify_and_build_rows(generate_problems())
    math_rows = [r for r in rows if r["metadata"]["domain"] == "math"]
    assert math_rows
    for row in math_rows:
        assert r"\boxed{" in row["messages"][1]["content"]


def test_code_rows_use_fenced_python() -> None:
    rows, _ = verify_and_build_rows(generate_problems())
    code_rows = [r for r in rows if r["metadata"]["domain"] == "code"]
    assert code_rows
    for row in code_rows:
        content = row["messages"][1]["content"]
        assert content.startswith("```python")
        assert "def " in content


def test_no_eval_families_in_curriculum() -> None:
    probs = generate_problems()
    eval_fams = {"derivative_chain", "integrate_func", "second_derivative"}
    for tier in probs:
        for p in probs[tier]["math"]:
            assert p["family"] not in eval_fams


def test_run_check_is_clean() -> None:
    code, result = run(check_only=True)
    assert code == 0
    assert result["contamination"]["clean"] is True


@pytest.mark.skipif(not OUT_DIR.joinpath("manifest.json").exists(), reason="curriculum not built")
def test_manifest_on_disk() -> None:
    manifest = json.loads((OUT_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "sophia.math_code_curriculum.v1"
    assert manifest["canClaimAGI"] is False
    assert manifest["contamination"]["clean"] is True
    assert manifest["counts"]["total"] > 0

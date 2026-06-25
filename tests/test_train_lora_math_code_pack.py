# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""train_lora.py accepts the math-code curriculum pack format."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_lora import MATH_CODE_PACK_DIR, load_rows, resolve_train_path  # noqa: E402
PACK = ROOT / "training" / "sophia-math-code-curriculum"
SFT = PACK / "sft_all.jsonl"


@pytest.mark.skipif(not SFT.exists(), reason="curriculum pack not built")
def test_resolve_pack_directory_to_sft_all() -> None:
    assert resolve_train_path(PACK) == SFT


@pytest.mark.skipif(not SFT.exists(), reason="curriculum pack not built")
def test_pack_rows_have_messages_schema() -> None:
    rows = load_rows(SFT)
    assert len(rows) == 144
    row = rows[0]
    assert "messages" in row
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][1]["role"] == "assistant"
    meta = row["metadata"]
    assert meta["source"] == "sophia-math-code-curriculum"
    assert meta["trainingOracleOnly"] is True


@pytest.mark.skipif(not (PACK / "manifest.json").exists(), reason="manifest missing")
def test_manifest_matches_row_count() -> None:
    manifest = json.loads((PACK / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "sophia.math_code_curriculum.v1"
    assert manifest["counts"]["total"] == len(load_rows(SFT))
    assert manifest["canClaimAGI"] is False


def test_math_code_pack_dir_constant() -> None:
    assert MATH_CODE_PACK_DIR.name == "sophia-math-code-curriculum"

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the low-RAM certification glue (non-torch logic)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import certify_lowram  # noqa: E402


def test_certify_lowram_offline_invariants() -> None:
    pytest.importorskip("numpy")
    ok, detail = certify_lowram.offline_invariants()
    assert ok, detail["checks"]


def test_eval_loader_handles_messages_text_and_skips(tmp_path) -> None:
    p = tmp_path / "holdout.jsonl"
    p.write_text("\n".join([
        json.dumps({"messages": [{"role": "user", "content": "a" * 40},
                                 {"role": "assistant", "content": "b" * 40}]}),
        json.dumps({"text": "c" * 40}),
        json.dumps({"text": "short"}),     # < min_chars -> skipped
        "",                                # blank -> skipped
        "{not json}",                      # unparseable -> skipped
    ]), encoding="utf-8")
    texts = certify_lowram.load_eval_texts(p, max_rows=10)
    assert len(texts) == 2
    assert all(len(t) >= 16 for t in texts)
    assert certify_lowram.load_eval_texts(p, max_rows=1) == texts[:1]

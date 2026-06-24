#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the external-oracle eval (agent/external_eval.py). Offline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import external_eval as e  # noqa: E402

ITEMS = [
    {"id": "a", "question": "2+2?", "answer": "4"},
    {"id": "b", "question": "10-3?", "answer": "7"},
    {"id": "c", "question": "eggs", "answer": "#### 39"},
]


def test_extract_answer() -> None:
    assert e.extract_answer("the answer is 42") == 42
    assert e.extract_answer("blah #### 39 done") == 39
    assert e.extract_answer("1,200 items") == 1200
    assert e.extract_answer("no number") is None


def test_score_item_uses_gold() -> None:
    assert e.score_item({"answer": "4"}, "I think it's 4.") is True
    assert e.score_item({"answer": "4"}, "I think it's 5.") is False
    assert e.score_item({"answer": "#### 39"}, "final: 39") is True


def test_run_dataset_accuracy() -> None:
    perfect = e.run_dataset(ITEMS, lambda it: str(it["answer"]))
    assert perfect["accuracy"] == 1.0 and perfect["correct"] == 3
    wrong = e.run_dataset(ITEMS, lambda it: "0")
    assert wrong["accuracy"] == 0.0


def test_committed_sample_loads_and_is_consistent() -> None:
    path = ROOT / "eval" / "external" / "gsm8k-style-sample.jsonl"
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(items) >= 10
    # the sample's own gold is internally consistent (perfect solver scores 100%)
    assert e.run_dataset(items, lambda it: str(it["answer"]))["accuracy"] == 1.0


def main() -> int:
    test_extract_answer()
    test_score_item_uses_gold()
    test_run_dataset_accuracy()
    test_committed_sample_loads_and_is_consistent()
    print("test_external_eval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

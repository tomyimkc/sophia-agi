#!/usr/bin/env python3
"""Offline tests for the external-dataset fetcher's CONVERSION (no network).

The download path needs the internet and is not run in CI; the parsing/reshaping
is a pure function and is fully tested here, including that its output is scored
correctly by the external-eval extractor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import external_eval  # noqa: E402
from tools.fetch_eval_dataset import convert_gsm8k_lines  # noqa: E402


def test_converts_valid_skips_malformed() -> None:
    lines = [
        json.dumps({"question": "2+3?", "answer": "two plus three\n#### 5"}),
        "",                                   # blank → skip
        "{not json",                          # malformed → skip
        json.dumps({"question": "", "answer": "#### 9"}),     # empty question → skip
        json.dumps({"question": "10/2?", "answer": "#### 5"}),
    ]
    items = convert_gsm8k_lines(lines)
    assert len(items) == 2
    assert all(set(it) == {"question", "answer"} for it in items)


def test_output_is_scorable_by_external_eval() -> None:
    lines = [json.dumps({"question": "2+3?", "answer": "work\n#### 5"})]
    item = convert_gsm8k_lines(lines)[0]
    # a correct answer scores True; a wrong one scores False — gold read from #### marker
    assert external_eval.score_item(item, "the answer is #### 5") is True
    assert external_eval.score_item(item, "the answer is 6") is False


def main() -> int:
    test_converts_valid_skips_malformed()
    test_output_is_scorable_by_external_eval()
    print("test_fetch_eval_dataset: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

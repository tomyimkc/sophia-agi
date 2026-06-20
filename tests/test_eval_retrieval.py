#!/usr/bin/env python3
"""Test for the retrieval eval harness (offline, real corpus)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_retrieval as er  # noqa: E402


def test_evaluate_shape_and_signal() -> None:
    report = er.evaluate(er.GOLDEN, top_k=8)
    assert report["queryCount"] == len(er.GOLDEN)
    assert set(report["recallAtK"]) == {"@1", "@3", "@5"}
    assert 0.0 <= report["mrr"] <= 1.0
    # the retriever should surface at least one expected source within top-5
    assert report["recallAtK"]["@5"] > 0.0


def test_custom_golden() -> None:
    golden = [{"query": "source discipline provenance attribution", "expect": "data/"}]
    report = er.evaluate(golden, top_k=8)
    assert report["queryCount"] == 1


def main() -> int:
    test_evaluate_shape_and_signal()
    test_custom_golden()
    print("test_eval_retrieval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

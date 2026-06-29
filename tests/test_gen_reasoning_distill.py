#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-gated reasoning distillation: only gate-clean teacher traces become SFT rows.

Deterministic, offline — the real gate, no model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.gen_reasoning_distill import (  # noqa: E402
    SELF_TEST_TRACES,
    row_from_trace,
    run,
    self_test,
)


def test_self_test_passes() -> None:
    assert self_test() == 0


def test_clean_trace_becomes_think_delimited_row() -> None:
    row, reason = row_from_trace(SELF_TEST_TRACES[0])
    assert reason is None and row is not None
    content = row["messages"][1]["content"]
    assert content.startswith("<think>") and "</think>" in content
    assert row["metadata"]["label_source"] == "machine_verified"
    assert row["metadata"]["verified"] is True


def test_hallucinated_answer_is_dropped() -> None:
    row, reason = row_from_trace(SELF_TEST_TRACES[1])
    assert row is None
    assert reason.startswith("answer_failed_gate")


def test_missing_fields_dropped() -> None:
    assert row_from_trace({"prompt": "", "thinking": "x", "answer": "y"})[0] is None
    assert row_from_trace({"prompt": "p", "answer": "a"})[1] == "missing_thinking"


def test_run_counts_kept_and_dropped() -> None:
    rows, stats = run(SELF_TEST_TRACES)
    assert stats["traces"] == 2 and stats["kept"] == 1 and stats["dropped"] == 1
    assert len(rows) == 1


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} gen_reasoning_distill tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Held-out prompt-quality validation: the predicate must separate the dev pack.

Deterministic, offline. NOTE: heldout_v1 is the development pack the detectors were tuned
against; floor-met here is self-consistency, not generalization (an independent v2 pack is OPEN).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_prompt_quality import MIN_PRECISION, MIN_RECALL, evaluate, offline_invariants  # noqa: E402


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_pack_has_both_labels_and_size() -> None:
    res = evaluate()
    assert res["n"] >= 20
    assert (res["tp"] + res["fn"]) >= 8   # enough positives
    assert (res["tn"] + res["fp"]) >= 8   # enough negatives


def test_predicate_separates_dev_pack_at_floor() -> None:
    res = evaluate()
    assert res["precision"] >= MIN_PRECISION, res["errors"]
    assert res["recall"] >= MIN_RECALL, res["errors"]


def test_result_carries_honest_caveat() -> None:
    # The harness must not silently claim a validated gate on the pack it was tuned against.
    res = evaluate()
    assert "caveat" in res
    assert "OPEN" in res["caveat"]
    assert res["verdict"] in ("FLOOR-MET", "FLOOR-UNMET")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} eval_prompt_quality tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

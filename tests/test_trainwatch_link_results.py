#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Linker that mirrors non-training result JSONs into TrainWatch — extractor is pure/offline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.trainwatch_link_results import _MAX_METRICS, extract_run  # noqa: E402


def test_verdict_booleans_become_numeric() -> None:
    r = extract_run({"passed": True, "validated": False, "top1_agreement": 0.95}, "result:x")
    assert r["metrics"]["passed"] == 1.0 and r["metrics"]["validated"] == 0.0
    assert r["metrics"]["top1_agreement"] == 0.95
    assert r["status"] == "completed" and r["name"] == "result:x"


def test_non_numeric_dropped_and_capped() -> None:
    res = {"scheme": "nvfp4", "note": "x", **{f"m{i}": i for i in range(20)}}
    r = extract_run(res, "result:y")
    assert "scheme" not in r["metrics"] and "note" not in r["metrics"]
    assert len(r["metrics"]) == _MAX_METRICS


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} trainwatch_link_results tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

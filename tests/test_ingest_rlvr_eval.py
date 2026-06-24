#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.ingest_rlvr_eval import ingest, map_report  # noqa: E402


def _report(before, after, base_fp, adapter_fp, entity=None, adapter="ckpt/sophia-rlvr-v1") -> dict:
    return {
        "benchmark": "rlvr-adapter-heldout", "adapter": adapter,
        "base": {"meanReward": before, "trueFalsePositiveRate": base_fp},
        "adapterScore": {"meanReward": after, "trueFalsePositiveRate": adapter_fp},
        "entityIntersection": entity or [],
    }


def _write(d: dict) -> str:
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(d, f)
    f.close()
    return f.name


def test_clean_improving_adapter_promotes() -> None:
    rec = ingest(_write(_report(0.71, 0.80, 0.02, 0.02)))
    assert rec["verdict"] == "promote"


def test_false_positive_regression_rejects() -> None:
    # FP rate rises 0.02 -> 0.10 => integrity 0.98 -> 0.90 => protected regression.
    rec = ingest(_write(_report(0.71, 0.85, 0.02, 0.10)))
    assert rec["verdict"] == "reject"


def test_contamination_rejects() -> None:
    rec = ingest(_write(_report(0.71, 0.90, 0.02, 0.02, entity=["entityA"])))
    assert rec["verdict"] == "reject"


def test_missing_fp_rate_fails_closed() -> None:
    # No trueFalsePositiveRate -> unverified integrity must NOT be treated as perfect.
    bad = {
        "adapter": "ckpt/x",
        "base": {"meanReward": 0.71},
        "adapterScore": {"meanReward": 0.85},
        "entityIntersection": [],
    }
    try:
        map_report(bad)
        raised = False
    except SystemExit:
        raised = True
    assert raised, "expected fail-closed error when FP rate is missing"


def test_missing_before_after_errors() -> None:
    bad = {"benchmark": "x", "rewards": {"trueGood": {}}}  # training report, no base/adapterScore
    try:
        map_report(bad)
        raised = False
    except SystemExit:
        raised = True
    assert raised, "expected a clear error when before/after pair is absent"


def test_mapping_inverts_fp_to_integrity() -> None:
    m = map_report(_report(0.7, 0.8, 0.05, 0.01))
    assert m["protected_before"] == 0.95 and m["protected_after"] == 0.99
    assert m["id"] == "sophia-rlvr-v1"  # basename of adapter path


def main() -> int:
    test_clean_improving_adapter_promotes()
    test_false_positive_regression_rejects()
    test_contamination_rejects()
    test_missing_fp_rate_fails_closed()
    test_missing_before_after_errors()
    test_mapping_inverts_fp_to_integrity()
    print("test_ingest_rlvr_eval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
    # passAt1 is the load-bearing capability metric; meanReward rides along as the
    # advisory aggregate (deliberately DIFFERENT values so a test can prove the gate
    # read passAt1, not meanReward).
    return {
        "benchmark": "rlvr-adapter-heldout", "adapter": adapter,
        "base": {"passAt1": before, "meanReward": round(before - 0.1, 4),
                 "trueFalsePositiveRate": base_fp},
        "adapterScore": {"passAt1": after, "meanReward": round(after - 0.1, 4),
                         "trueFalsePositiveRate": adapter_fp},
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
        "base": {"passAt1": 0.71},
        "adapterScore": {"passAt1": 0.85},
        "entityIntersection": [],
    }
    try:
        map_report(bad)
        raised = False
    except SystemExit:
        raised = True
    assert raised, "expected fail-closed error when FP rate is missing"


def test_gates_on_pass_at1_not_mean_reward() -> None:
    # The gate must read passAt1 (load-bearing), NOT the advisory meanReward
    # (which in this fixture is deliberately offset by -0.1 on both sides).
    m = map_report(_report(0.51, 0.53, 0.02, 0.02))
    assert m["capabilityMetric"] == "passAt1"
    assert m["before"] == 0.51 and m["after"] == 0.53
    assert m["meanRewardAdvisory"] == {"before": 0.41, "after": 0.43}


def test_missing_pass_at1_fails_closed_never_falls_back_to_mean_reward() -> None:
    # A meanReward-only report (the run66/run70 sweep shape) must be REFUSED with
    # reason 'passAt1 missing' — never silently gated on meanReward.
    bad = {
        "adapter": "ckpt/sophia-rlvr-v1",
        "base": {"meanReward": 0.5819, "trueFalsePositiveRate": 0.1304},
        "adapterScore": {"meanReward": 0.7819, "trueFalsePositiveRate": 0.1304},
        "entityIntersection": [],
    }
    try:
        map_report(bad)
        raised = False
        msg = ""
    except SystemExit as e:
        raised = True
        msg = str(e)
    assert raised, "expected fail-closed refusal when passAt1 is absent"
    assert "passAt1 missing" in msg


def test_missing_adapter_side_pass_at1_fails_closed() -> None:
    # passAt1 present on base only -> still refused (both sides required).
    bad = _report(0.51, 0.53, 0.02, 0.02)
    del bad["adapterScore"]["passAt1"]
    try:
        map_report(bad)
        raised = False
        msg = ""
    except SystemExit as e:
        raised = True
        msg = str(e)
    assert raised and "passAt1 missing" in msg


def test_mean_reward_advisory_fail_open_when_absent() -> None:
    # Advisory aggregate absent (e.g. the invention task emits no meanReward):
    # ingest must still map cleanly on passAt1, with None advisories.
    rep = _report(0.51, 0.71, 0.02, 0.02)
    del rep["base"]["meanReward"]
    del rep["adapterScore"]["meanReward"]
    m = map_report(rep)
    assert m["before"] == 0.51 and m["after"] == 0.71
    assert m["meanRewardAdvisory"] == {"before": None, "after": None}


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
    test_gates_on_pass_at1_not_mean_reward()
    test_missing_pass_at1_fails_closed_never_falls_back_to_mean_reward()
    test_missing_adapter_side_pass_at1_fails_closed()
    test_mean_reward_advisory_fail_open_when_absent()
    test_missing_before_after_errors()
    test_mapping_inverts_fp_to_integrity()
    print("test_ingest_rlvr_eval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the AATS benchmark (tools/aats_benchmark.py).

Deterministic, offline, dependency-free. Asserts the load-bearing claims the benchmark
characterises hold, and that the benchmark is reproducible.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.aats_benchmark import benchmark_ensemble, run  # noqa: E402

_REP = run()


def test_all_load_bearing_checks_pass():
    assert _REP["allChecksPass"] is True, _REP["loadBearingChecks"]


def test_ensemble_spans_all_quadrants_at_scale():
    e = _REP["ensemble"]
    assert e["nBad"] >= 100                              # powered, not a toy
    for q in ("both", "temporal-only", "provenance-only", "neither"):
        assert e["quadrantsCaught"][q] > 0, q           # every catch-quadrant populated


def test_consensus_catches_more_and_is_significant():
    e = _REP["ensemble"]
    mc = e["mcnemarConsensusVsBestSingle"]
    assert mc["c"] > mc["b"]                             # consensus catches strictly more
    assert mc["p"] < 0.05                                # and significantly so (powered)
    # AND-consensus false-approval <= the best single verifier's
    best = e["bestSingle"]
    assert e["andConsensus"]["falseApprovalRate"] <= e["perVerifier"][best]["falseApprovalRate"]


def test_consensus_honestly_misses_the_neither_quadrant():
    # consensus is not magic: misattributions NEITHER family can catch are still false-approved
    e = _REP["ensemble"]
    assert e["quadrantsCaught"]["neither"] > 0
    assert e["andConsensus"]["falseApprovalRate"] > 0.0


def test_conformal_within_bucket_validity_holds_but_naive_degrades():
    c = _REP["conformal"]
    assert c["withinBucketValidityHeldRate"] >= 0.95     # guarantee holds when used correctly
    assert c["naiveCrossBucketValidityHeldRate"] < c["withinBucketValidityHeldRate"]  # the finding
    assert c["priceMonotone"] is True


def test_breaker_detection_complete():
    assert _REP["breaker"]["detectionComplete"] is True


def test_benchmark_is_deterministic():
    a = benchmark_ensemble()
    b = benchmark_ensemble()
    assert a["andConsensus"] == b["andConsensus"]
    assert a["mcnemarConsensusVsBestSingle"] == b["mcnemarConsensusVsBestSingle"]
    assert a["quadrantsCaught"] == b["quadrantsCaught"]


if __name__ == "__main__":
    failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"ok {_name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {_name}: {exc}")
    print("all passed" if not failures else f"{failures} FAILED")
    raise SystemExit(1 if failures else 0)

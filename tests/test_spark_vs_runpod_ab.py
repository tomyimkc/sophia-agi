# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline test for the Spark-vs-RunPod A/B divergence harness (Spark-6)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.spark_vs_runpod_ab import compare  # noqa: E402


def test_code_task_divergence() -> None:
    local = {"task": "code", "model": "m", "base": {"passAt1": 0.0}, "adapterScore": {"passAt1": 0.08}}
    runpod = {"task": "code", "model": "m", "base": {"passAt1": 0.0}, "adapterScore": {"passAt1": 0.10}}
    c = compare(local, runpod)
    assert c["metric"] == "passAt1"
    assert c["spark"]["delta"] == 0.08 and c["runpod"]["delta"] == 0.10
    assert c["divergence"]["adapterAbsolute"] == 0.02
    assert c["candidateOnly"] and c["canClaimAGI"] is False


def test_provenance_task_uses_mean_reward() -> None:
    local = {"task": "provenance", "base": {"meanReward": 0.4}, "adapterScore": {"meanReward": 0.6}}
    runpod = {"task": "provenance", "base": {"meanReward": 0.4}, "adapterScore": {"meanReward": 0.62}}
    c = compare(local, runpod)
    assert c["metric"] == "meanReward" and c["divergence"]["adapterAbsolute"] == 0.02


def test_task_mismatch_rejected() -> None:
    # code (passAt1) vs provenance (meanReward) is a real metric mismatch
    try:
        compare({"task": "code", "base": {"passAt1": 0}, "adapterScore": {"passAt1": 0.1}},
                {"task": "provenance", "base": {"meanReward": 0.4}, "adapterScore": {"meanReward": 0.6}})
    except ValueError:
        return
    raise AssertionError("metric/task mismatch should be rejected")


def test_code_vs_math_task_mismatch_rejected() -> None:
    # both code and math map to passAt1, but they are DIFFERENT tasks — must still reject
    # (comparing across tasks conflates the hardware gap with a task-content gap)
    try:
        compare({"task": "code", "base": {"passAt1": 0}, "adapterScore": {"passAt1": 0.1}},
                {"task": "math", "base": {"passAt1": 0}, "adapterScore": {"passAt1": 0.1}})
    except ValueError as exc:
        assert "task mismatch" in str(exc).lower()
        return
    raise AssertionError("code-vs-math task mismatch should be rejected (not just metric)")


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_spark_vs_runpod_ab: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

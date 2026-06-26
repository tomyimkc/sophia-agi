# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Inference-serving M0 — batching simulator tests (plain-script, stdlib)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.sim.engine import compare, simulate  # noqa: E402
from serving.sim.workload import Request, poisson_workload  # noqa: E402


def test_all_requests_finish() -> None:
    reqs = poisson_workload(20, rate=0.7, seed=1)
    for policy in ("static", "continuous"):
        r = simulate(reqs, policy=policy, max_batch=4)
        assert r.n == 20 and r.makespan > 0 and r.output_tokens > 0


def test_continuous_beats_static_under_load() -> None:
    # Bursty arrivals + varied output lengths → static head-of-line blocking.
    reqs = poisson_workload(24, rate=0.8, seed=0)
    s = simulate(reqs, policy="static", max_batch=4)
    c = simulate(reqs, policy="continuous", max_batch=4)
    assert c.throughput >= s.throughput
    assert c.ttft_mean <= s.ttft_mean


def test_metrics_bounds_and_determinism() -> None:
    reqs = poisson_workload(16, rate=0.6, seed=2)
    a = compare(reqs, max_batch=4)
    b = compare(reqs, max_batch=4)
    assert a == b  # deterministic
    assert 0.0 <= a["continuous"]["goodput"] <= 1.0
    assert a["throughputSpeedup"] >= 1.0


def test_single_request_ttft_tracks_prefill() -> None:
    # One request, arrives at t=0, prompt 32 (prefill_chunk 16 → 2 prefill ticks),
    # output 5. First token after prefill; finishes after output ticks.
    r = Request(rid=0, arrival=0, prompt_len=32, output_len=5)
    res = simulate([r], policy="continuous", max_batch=1, prefill_chunk=16)
    assert res.n == 1 and res.output_tokens == 5
    assert res.ttft_mean >= 2.0  # at least the 2 prefill ticks before first token


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"PASSED {len(tests)} serving-sim tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())

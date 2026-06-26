# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Kernels M0 — reference numerics + roofline classification (plain-script, stdlib)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels import reference as ref  # noqa: E402

DEVICE = "NVIDIA DGX Spark GB10"


def test_softmax_correct() -> None:
    out = ref.softmax([1.0, 2.0, 3.0])
    assert abs(sum(out) - 1.0) < 1e-12
    assert out[0] < out[1] < out[2]
    # Matches the closed form for a known input.
    z = [math.exp(v - 3.0) for v in (1.0, 2.0, 3.0)]
    s = sum(z)
    assert all(abs(a - b / s) < 1e-12 for a, b in zip(out, z))


def test_rmsnorm_correct() -> None:
    out = ref.rmsnorm([3.0, 4.0], [1.0, 1.0], eps=0.0)
    rms = math.sqrt((9 + 16) / 2)
    assert all(abs(o - v / rms) < 1e-9 for o, v in zip(out, (3.0, 4.0)))


def test_swiglu_matches_silu_times_up() -> None:
    out = ref.swiglu([0.0, 1.0, -2.0], [2.0, 3.0, 4.0])
    expect = [ref.silu(0.0) * 2.0, ref.silu(1.0) * 3.0, ref.silu(-2.0) * 4.0]
    assert all(abs(a - b) < 1e-12 for a, b in zip(out, expect))


def test_ops_are_memory_bound() -> None:
    # Elementwise/reduction ops have tiny arithmetic intensity << ridge point,
    # so fusion (fewer HBM round-trips) is the win.
    ridge = ref.ridge_point(DEVICE)
    for op in ("softmax", "rmsnorm", "swiglu"):
        c = ref.classify(op, 4096, 4096, DEVICE)
        assert c["intensity"] < ridge
        assert c["regime"] == "memory-bound"


def test_op_cost_accounting() -> None:
    c = ref.op_cost("swiglu", 2, 8)   # 2 inputs read + 1 output write
    assert c["flops"] == 6 * 16
    assert c["bytes"] == 2 * (2 * 16 + 16)


def test_run_kernels_offline_invariants() -> None:
    import importlib
    mod = importlib.import_module("tools.run_kernels")
    ok, detail = mod._offline_invariants(DEVICE)
    assert ok, [k for k, v in detail["checks"].items() if not v]


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"PASSED {len(tests)} kernels-reference tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())

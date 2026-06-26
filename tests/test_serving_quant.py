# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for expert offloading + KV cache quantization."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from serving import expert_offload, kv_quant  # noqa: E402


# ---- expert offloading -----------------------------------------------------

def test_expert_offload_offline_invariants() -> None:
    ok, detail = expert_offload.offline_invariants()
    assert ok, detail["checks"]


def test_only_routed_experts_resident() -> None:
    s = expert_offload.TieredExpertStore(gpu_budget_bytes=300)
    for e in range(8):
        s.register(e, size_bytes=100, tier=expert_offload.ExpertTier.DISK)
    s.route_select([0, 1])
    assert set(s.gpu_resident()) == {0, 1}
    assert all(s.tier_of(e) == expert_offload.ExpertTier.DISK for e in range(2, 8))


def test_gpu_budget_never_exceeded() -> None:
    s = expert_offload.TieredExpertStore(gpu_budget_bytes=250)
    for e in range(6):
        s.register(e, size_bytes=100, tier=expert_offload.ExpertTier.DISK)
    s.route_select([0, 1, 2, 3])  # would need 400 > 250
    assert s.stats.gpu_resident_bytes <= 250


def test_reroute_is_gpu_hit() -> None:
    s = expert_offload.TieredExpertStore(gpu_budget_bytes=500)
    s.register(0, size_bytes=100, tier=expert_offload.ExpertTier.DISK)
    s.route_select([0])
    hits = s.stats.gpu_hits
    s.route_select([0])
    assert s.stats.gpu_hits == hits + 1


# ---- KV cache quantization -------------------------------------------------

def test_kv_quant_offline_invariants() -> None:
    ok, detail = kv_quant.offline_invariants()
    assert ok, detail["checks"]


def test_int8_kv_error_bounded() -> None:
    rng = np.random.default_rng(0)
    KV = rng.standard_normal((32, 16))
    q, scale = kv_quant.quantize_kv_block(KV, bits=8)
    dq = kv_quant.dequantize_kv_block(q, scale)
    assert np.max(np.abs(dq - KV)) <= scale / 2 + 1e-12


def test_identical_content_identical_quant() -> None:
    """Prefix-sharing safety: same tokens → same quantized block."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal((16, 8))
    qa, sa = kv_quant.quantize_kv_block(a, bits=8)
    qb, sb = kv_quant.quantize_kv_block(a.copy(), bits=8)
    assert np.array_equal(qa, qb) and sa == sb


def test_memory_ratios() -> None:
    assert kv_quant.kv_memory_ratio(16, 8) == 2.0   # INT8 vs FP16
    assert kv_quant.kv_memory_ratio(16, 4) == 4.0   # INT4 vs FP16

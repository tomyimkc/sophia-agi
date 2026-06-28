# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for AirLLM-style layer streaming + the low-RAM measurement gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving import layer_stream, lowram_eval  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import shard_checkpoint  # noqa: E402


# ---- layer streaming -------------------------------------------------------

def test_layer_stream_offline_invariants() -> None:
    ok, detail = layer_stream.offline_invariants()
    assert ok, detail["checks"]


def test_full_pass_stays_within_budget() -> None:
    """The low-RAM guarantee: peak GPU bytes << whole model over a full forward pass."""
    s = layer_stream.StreamingLayerStore(gpu_budget_bytes=300, prefetch_depth=1)
    for i in range(40):
        s.register(i, fp16_bytes=100, tier=layer_stream.LayerTier.DISK)
    for _ in s.forward_pass(40):
        pass
    assert s.stats.peak_gpu_bytes <= 300
    assert s.stats.peak_gpu_bytes < 40 * 100


def test_stream_once_no_thrash() -> None:
    """A window wide enough for the prefetch depth loads each layer from disk exactly once."""
    s = layer_stream.StreamingLayerStore(gpu_budget_bytes=400, prefetch_depth=1)
    for i in range(20):
        s.register(i, fp16_bytes=100, tier=layer_stream.LayerTier.DISK)
    for _ in s.forward_pass(20):
        pass
    assert s.stats.disk_loads == 20


def test_prefetch_creates_hits() -> None:
    s = layer_stream.StreamingLayerStore(gpu_budget_bytes=400, prefetch_depth=1)
    for i in range(10):
        s.register(i, fp16_bytes=100, tier=layer_stream.LayerTier.DISK)
    for _ in s.forward_pass(10):
        pass
    assert s.stats.prefetch_hits >= 9  # every layer but the first


def test_quant_aware_sizing_holds_more_layers() -> None:
    fp16 = layer_stream.StreamingLayerStore(gpu_budget_bytes=320, prefetch_depth=4)
    q = layer_stream.StreamingLayerStore(gpu_budget_bytes=320, prefetch_depth=4)
    for i in range(16):
        fp16.register(i, fp16_bytes=100, bits=16)
        q.register(i, fp16_bytes=100, bits=4)
    fp16.step(0)
    q.step(0)
    assert len(q.gpu_resident()) > len(fp16.gpu_resident())


def test_resident_bytes_for_ratio() -> None:
    assert layer_stream.resident_bytes_for(100, 16) == 100
    assert layer_stream.resident_bytes_for(100, 8) == 50
    assert layer_stream.resident_bytes_for(100, 4) == 25
    with pytest.raises(ValueError):
        layer_stream.resident_bytes_for(100, 0)


def test_plan_layer_bits_hits_target() -> None:
    pytest.importorskip("numpy")
    fp16 = {i: 1000 for i in range(8)}
    bits = layer_stream.plan_layer_bits(fp16, target_avg_bits=4.0, protected={0, 7})
    assert all(1 <= b <= 16 for b in bits.values())
    assert bits[0] >= 6 and bits[7] >= 6          # protected floor honored
    avg = sum(bits.values()) / len(bits)
    assert 3.0 <= avg <= 6.0


# ---- low-RAM measurement gate ----------------------------------------------

def test_lowram_eval_offline_invariants() -> None:
    pytest.importorskip("numpy")
    ok, detail = lowram_eval.offline_invariants()
    assert ok, detail["checks"]


def test_identical_model_passes_gate() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    z = rng.standard_normal((32, 40))
    e = np.exp(z - z.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    rep = lowram_eval.LowRamGate().evaluate(p, p.copy(), mem_ratio=3.56)
    assert rep.passed and rep.mean_kl < 1e-9 and rep.top1_agreement == 1.0


def test_degraded_model_fails_gate() -> None:
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    z = rng.standard_normal((32, 40)) * 3
    e = np.exp(z - z.max(1, keepdims=True))
    full = e / e.sum(1, keepdims=True)
    z2 = z + rng.standard_normal((32, 40)) * 6
    e2 = np.exp(z2 - z2.max(1, keepdims=True))
    low = e2 / e2.sum(1, keepdims=True)
    assert not lowram_eval.LowRamGate().evaluate(full, low).passed


# ---- checkpoint sharding ---------------------------------------------------

def test_shard_checkpoint_offline_invariants() -> None:
    ok, detail = shard_checkpoint.offline_invariants()
    assert ok, detail["checks"]


def test_plan_one_shard_per_layer(tmp_path) -> None:
    import json
    cfg = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 12,
           "hidden_size": 512, "intermediate_size": 1376, "num_attention_heads": 8,
           "num_key_value_heads": 8, "vocab_size": 32000, "torch_dtype": "bfloat16"}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    plan = shard_checkpoint.build_plan(tmp_path / "config.json", target_avg_bits=None)
    decoder = [s for s in plan["shards"] if s["kind"] == "decoder_layer"]
    assert len(decoder) == 12
    assert [s["layer_index"] for s in decoder] == list(range(12))
    assert plan["est_layer_bytes"] > 0

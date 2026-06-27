# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""LoRA rank allocation (moe/adapt) — the P6 moe/adapt -> QLoRA bridge.

Verifies that lora_rank_allocation reuses the bit_allocator's budget/floor/range/ordering
guarantees with rank semantics, so it can drive a PEFT rank_pattern without exceeding the
uniform-rank parameter budget.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from moe import adapt  # noqa: E402


def _modules():
    # (name, fan_in, fan_out, sensitivity, protected) for a Qwen-style attn+mlp block.
    return [
        ("q_proj", 2048, 2048, 4.0, True),
        ("k_proj", 2048, 256, 2.0, False),
        ("v_proj", 2048, 256, 4.0, True),
        ("o_proj", 2048, 2048, 3.0, False),
        ("gate_proj", 2048, 8192, 1.5, False),
        ("up_proj", 2048, 8192, 1.0, False),
        ("down_proj", 8192, 2048, 1.5, False),
    ]


def test_offline_invariants_pass():
    ok, detail = adapt.offline_invariants()
    assert ok, detail["checks"]


def test_budget_preserved_vs_uniform():
    """Allocated adapter params must not exceed the uniform-rank=r budget."""
    mods = _modules()
    r = 8
    rank_pattern, _ = adapt.lora_rank_allocation(
        mods, target_avg_rank=r, min_rank=4, max_rank=32, protected_rank=8)
    cost = {m[0]: m[1] + m[2] for m in mods}
    alloc = sum(rank_pattern[n] * cost[n] for n in rank_pattern)
    uniform = r * sum(cost.values())
    assert alloc <= uniform + 1e-6


def test_protected_floor_and_range():
    mods = _modules()
    rank_pattern, _ = adapt.lora_rank_allocation(
        mods, target_avg_rank=8, min_rank=4, max_rank=32, protected_rank=8)
    assert rank_pattern["q_proj"] >= 8 and rank_pattern["v_proj"] >= 8  # protected
    assert all(4 <= r <= 32 for r in rank_pattern.values())


def test_sensitivity_redistribution():
    """Sensitive attention modules get >= the redundant MLP modules."""
    mods = _modules()
    rp, _ = adapt.lora_rank_allocation(
        mods, target_avg_rank=8, min_rank=4, max_rank=32, protected_rank=8)
    assert rp["q_proj"] >= rp["up_proj"]
    assert rp["v_proj"] >= rp["down_proj"]


def test_alpha_tracks_rank():
    mods = _modules()
    rp, ap = adapt.lora_rank_allocation(
        mods, target_avg_rank=8, alpha_per_rank=2.0, min_rank=4, max_rank=32, protected_rank=8)
    assert ap.keys() == rp.keys()
    assert all(ap[n] == round(2.0 * rp[n]) for n in rp)


def test_empty_modules_safe():
    assert adapt.lora_rank_allocation([], target_avg_rank=8) == ({}, {})


def test_weight_norm_sensitivity_proxy():
    rng = np.random.default_rng(0)
    big = rng.standard_normal((64, 64)) * 5.0
    small = rng.standard_normal((64, 64)) * 0.1
    sens = adapt.weight_norm_sensitivity({"big": big, "small": small})
    assert sens["big"] > sens["small"] > 0.0

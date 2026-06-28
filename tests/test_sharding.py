# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for FSDP + expert-parallel sharding (high/top-tier multi-GPU training)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training import sharding  # noqa: E402

DS_V3 = dict(total_params=671_000_000_000, embed_params=2 * 129280 * 7168,
             n_routed_experts=256, n_layers=61)


def test_sharding_offline_invariants() -> None:
    ok, detail = sharding.offline_invariants()
    assert ok, detail["checks"]


def test_expert_assignment_is_balanced_partition() -> None:
    asn = sharding.expert_assignment(160, 8)
    flat = [e for r in asn for e in r]
    assert sorted(flat) == list(range(160))           # every expert owned once
    assert max(len(r) for r in asn) - min(len(r) for r in asn) <= 1


def test_671b_qlora_needs_sharding_fits_8x80gb() -> None:
    one = sharding.plan_sharding(world_size=1, base_bits=4.0, fits_gpu_gb=80.0, activation_gb=4.0, **DS_V3)
    eight = sharding.plan_sharding(world_size=8, base_bits=4.0, fits_gpu_gb=80.0, activation_gb=4.0, **DS_V3)
    assert not one.fits                                # 671B does not fit one GPU
    assert eight.fits                                  # but shards to fit 8×80GB at 4-bit
    assert eight.per_rank_gb < one.per_rank_gb


def test_bits_lever_bf16_needs_more_ranks() -> None:
    w4 = sharding.min_world_size_to_fit(base_bits=4.0, fits_gpu_gb=80.0, activation_gb=4.0, **DS_V3)
    w16 = sharding.min_world_size_to_fit(base_bits=16.0, fits_gpu_gb=80.0, activation_gb=4.0, **DS_V3)
    assert w16 > w4


def test_all_to_all_zero_single_rank_positive_multi() -> None:
    assert sharding.plan_sharding(world_size=1, base_bits=4.0, **DS_V3).all_to_all_tokens == 0
    assert sharding.plan_sharding(world_size=8, base_bits=4.0, **DS_V3).all_to_all_tokens > 0


def test_fail_closed() -> None:
    with pytest.raises(ValueError):
        sharding.expert_assignment(8, 0)
    with pytest.raises(ValueError):
        sharding.plan_sharding(world_size=0, base_bits=4.0, **DS_V3)
    with pytest.raises(ValueError):
        sharding.plan_sharding(world_size=2, base_bits=0, **DS_V3)

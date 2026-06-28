# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the low-RAM serving runtime + frontier RAM planner."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving import lowram_runtime as lr  # noqa: E402


def test_lowram_runtime_offline_invariants() -> None:
    ok, detail = lr.offline_invariants()
    assert ok, detail["checks"]


def test_glm_total_collapses_to_small_resident() -> None:
    rep = lr.plan_ram(lr.GLM_5_2)
    ops = rep["operating_points"]
    assert ops["dense_fp16"]["resident_gb"] > 1000          # ~1.49 TB dense
    assert ops["expert_offload"]["resident_gb"] < 64        # fits one GPU/Spark
    assert ops["full_stream"]["resident_gb"] < 10           # fits a small device
    assert rep["reduction_vs_dense"]["expert_offload"] > 50


def test_sparsity_ratio_is_the_lever() -> None:
    assert lr.GLM_5_2.sparsity_ratio > 15
    assert lr.DEEPSEEK_V3.sparsity_ratio > 15
    # A dense spec has ratio 1.0.
    dense = lr.ModelSpec(name="dense-7b", n_layers=32, hidden=4096, vocab=32000,
                         total_params=7_000_000_000, active_params=7_000_000_000)
    assert dense.sparsity_ratio == 1.0
    assert not dense.is_moe


def test_quant_width_monotone() -> None:
    # Lower weight bits → less resident RAM.
    hi = lr.plan_ram(lr.GLM_5_2, weight_bits=8.0)["operating_points"]["expert_offload"]["resident_gb"]
    lo = lr.plan_ram(lr.GLM_5_2, weight_bits=4.5)["operating_points"]["expert_offload"]["resident_gb"]
    assert lo < hi


def test_bad_weight_bits_rejected() -> None:
    with pytest.raises(ValueError):
        lr.plan_ram(lr.GLM_5_2, weight_bits=0)
    with pytest.raises(ValueError):
        lr.plan_ram(lr.GLM_5_2, weight_bits=32)


def test_runtime_composition_within_budget() -> None:
    spec = lr.ModelSpec(name="glm-nano", n_layers=12, hidden=512, vocab=4096,
                        total_params=1_000_000_000, active_params=120_000_000,
                        n_routed_experts=16, active_experts=2)
    rt = lr.LowRamRuntime(spec, gpu_budget_bytes=8_000_000, weight_bits=4.5, prefetch_depth=1)
    rt.decode_step()
    assert rt.peak_resident_bytes() <= rt.gpu_budget_bytes() + 1
    assert rt.trunk.stats.disk_loads >= spec.n_layers      # streamed every layer
    assert rt.experts.stats.promotes > 0                   # offloaded/promoted experts


def test_runtime_rejects_bad_expert_frac() -> None:
    spec = lr.SOPHIA_V1_TARGET
    with pytest.raises(ValueError):
        lr.LowRamRuntime(spec, gpu_budget_bytes=10_000_000, expert_budget_frac=0.0)
    with pytest.raises(ValueError):
        lr.LowRamRuntime(spec, gpu_budget_bytes=10_000_000, expert_budget_frac=1.0)


def test_sophia_target_is_frontier_total_low_resident() -> None:
    rep = lr.plan_ram(lr.SOPHIA_V1_TARGET)
    assert lr.SOPHIA_V1_TARGET.total_params >= 700_000_000_000
    assert rep["operating_points"]["full_stream"]["resident_gb"] < 10
    assert "NOT a capability claim" in rep["honest_scope"]


def test_reference_specs_exported() -> None:
    from serving import GLM_5_2, DEEPSEEK_V3, SOPHIA_V1_TARGET, plan_ram, LowRamRuntime, ModelSpec
    assert {GLM_5_2.name, DEEPSEEK_V3.name, SOPHIA_V1_TARGET.name} <= set(lr.REFERENCE_SPECS)

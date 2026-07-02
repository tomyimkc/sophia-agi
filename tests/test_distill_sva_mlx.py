# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/distill_sva_mlx.py (A2 SVA-lite math core; MLX step is bench-only)."""
from __future__ import annotations

import pytest

from tools.distill_sva_mlx import (
    domain_normalized_aggregate,
    offline_invariants,
    route_teacher,
    sva_position_loss,
    sva_sequence_loss,
    teacher_topk_support,
)


def test_offline_invariants_pass():
    ok, detail = offline_invariants()
    assert ok, detail


def test_topk_is_teacher_selected_and_deterministic():
    t = {"x": 0.4, "y": 0.4, "z": 0.2}
    assert teacher_topk_support(t, 2) == ["x", "y"]  # tie broken by token string
    with pytest.raises(ValueError):
        teacher_topk_support(t, 0)


def test_reverse_kl_direction_and_renormalization():
    teacher = {"a": 0.6, "b": 0.3, "c": 0.1}
    sharp_on_mode = {"a": 1.0}
    spread_off_mode = {"c": 0.9, "b": 0.1}
    l_mode, _ = sva_position_loss(sharp_on_mode, teacher, k=2)
    l_off, _ = sva_position_loss(spread_off_mode, teacher, k=2)
    # mode-seeking: concentrating on the teacher's top token costs less than
    # concentrating off-support/low-mode
    assert l_mode < l_off


def test_zero_support_mass_fails_closed_finite():
    teacher = {"a": 0.9, "b": 0.1}
    student = {"z": 1.0}  # no mass on teacher's support
    loss, rho = sva_position_loss(student, teacher, k=2)
    assert rho == 0.0 and loss > 10.0 and loss < float("inf")


def test_sequence_mask_and_domain_aggregation():
    t = {"a": 0.7, "b": 0.3}
    loss, rho, n = sva_sequence_loss([(t, t, True), (t, t, False)], k=2)
    assert n == 1 and loss == 0.0 and rho == pytest.approx(1.0)
    # Eq. 6: d2's single sample counts as much as d1's three
    agg = domain_normalized_aggregate([("d1", 2.0), ("d1", 2.0), ("d1", 2.0), ("d2", 0.0)])
    assert agg == pytest.approx(1.0)


def test_hard_routing_fails_closed():
    with pytest.raises(KeyError):
        route_teacher("philosophy", {"coding": object()})


def test_mlx_step_fails_closed_without_mlx():
    try:
        import mlx.core  # noqa: F401
        pytest.skip("mlx present; fail-closed path not exercised here")
    except ImportError:
        pass
    from tools.distill_sva_mlx import build_mlx_sva_step

    with pytest.raises(RuntimeError, match="requires mlx"):
        build_mlx_sva_step("Qwen/Qwen2.5-3B-Instruct", {}, "adapter")

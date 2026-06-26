# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the P7 router-policy ablation on fixed MoELM experts."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

np = pytest.importorskip("numpy")

from pretraining.architecture import p7_router_ablation as p7  # noqa: E402


def test_p7_offline_invariants() -> None:
    ok, detail = p7.offline_invariants()
    assert ok, detail["checks"]


def test_handrolled_matches_native_moe() -> None:
    """The hand-rolled-top1 policy must reproduce MoELM.forward exactly —
    it consumes the same argmax selection from the shared logits, so any
    delta is a bug, not a signal."""
    rep = p7.run_ablation(quick=True, seed=0)
    assert rep["self_consistency"]["handrolled_matches_native"] is True


def test_experts_frozen_after_routing() -> None:
    """Routing passes must not mutate the frozen expert weights — experts are
    read-only substrate, the only variable is the routing policy."""
    rep = p7.run_ablation(quick=True, seed=0)
    assert rep["self_consistency"]["experts_frozen"] is True


def test_nodrop_policy_equals_handrolled() -> None:
    """moerouter-top1-nodrop uses the identical selection as handrolled-top1
    (argmax, no drops) → held-out loss must be identical."""
    rep = p7.run_ablation(quick=True, seed=0)
    h = rep["policies"]["handrolled-top1"]["held_loss"]
    n = rep["policies"]["moerouter-top1-nodrop"]["held_loss"]
    assert abs(h - n) < 1e-3


def test_compute_equalized_to_moe_active() -> None:
    """The dense baseline's parameter count is tuned to match the MoE's ACTIVE
    param count (within 5%), unlike the prior qualitative-only matching."""
    rep = p7.run_ablation(quick=True, seed=0)
    d = rep["dense"]["params"]
    a = rep["moe_active_params"]
    assert abs(d - a) / max(1, a) < 0.05


def test_scope_present() -> None:
    """The honest-scope caveat carries the load-bearing phrase deferring any
    'superior router' claim to the no-overclaim gate."""
    rep = p7.run_ablation(quick=True, seed=0)
    assert p7.SCOPE_KEY.lower() in rep["honest_scope"].lower()


def test_deterministic_across_runs() -> None:
    """Same seed → identical rounded policy numbers and verdict."""
    a = p7.run_ablation(quick=True, seed=0)
    b = p7.run_ablation(quick=True, seed=0)
    assert a["policies"] == b["policies"]
    assert a["verdict"] == b["verdict"]

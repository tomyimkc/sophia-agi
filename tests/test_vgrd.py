# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for Verification-Gated Recurrent Depth (VGRD), the Phase-2 coupling.

Pure stdlib (no numpy/torch) — runs in the same lane as the nano study. Validates the
fail-closed accept/abstain/block policy and the selective-prediction machinery against a
controlled convergent/oscillating substrate with known ground truth.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pretraining.architecture import vgrd  # noqa: E402


def test_offline_invariants() -> None:
    ok, detail = vgrd.offline_invariants()
    assert ok, detail["checks"]


def test_accept_abstain_block_paths() -> None:
    """The three fail-closed verdicts on hand-built trajectories."""
    stable = [vgrd._logits_for(3, 6) for _ in range(8)]
    osc = [vgrd._logits_for(1 if t % 2 else 4, 6) for t in range(8)]
    assert vgrd.vgrd_decide(stable)["verdict"] == "accept"
    assert vgrd.vgrd_decide(osc)["verdict"] == "abstain"
    assert vgrd.vgrd_decide(stable, verify_fn=lambda a: False)["verdict"] == "block"


def test_depth_confidence_settles() -> None:
    """A trajectory that settles early reports high confidence and an early settle step;
    an oscillating one reports low confidence."""
    settled = [vgrd._logits_for(0, 5) for _ in range(6)]
    dc = vgrd.depth_confidence(settled)
    assert dc["confidence"] == 1.0 and dc["settle_step"] == 0 and dc["answer"] == 0
    osc = [vgrd._logits_for(t % 2, 5) for t in range(6)]
    assert vgrd.depth_confidence(osc)["confidence"] < 1.0


def test_abstention_lifts_selective_accuracy_and_kills_fabrication() -> None:
    """The headline coupling property: abstaining on the low-confidence tail raises
    selective accuracy and drives fabrication on the unanswerable set to zero."""
    rep = vgrd.run_study(quick=True, seed=0)
    rows = rep["selective_prediction"]["rows"]
    assert rows[-1]["selective_accuracy"] >= rows[0]["selective_accuracy"]
    assert rows[-1]["fabrication_rate_on_unanswerable"] == 0.0
    assert rep["accepted_fabrications_on_unanswerable"] == 0


def test_empty_trajectory_is_fail_closed() -> None:
    assert vgrd.vgrd_decide([])["verdict"] == "abstain"


def test_deterministic() -> None:
    assert vgrd.run_study(quick=True, seed=0) == vgrd.run_study(quick=True, seed=0)


def test_scope_present() -> None:
    rep = vgrd.run_study(quick=True, seed=0)
    assert vgrd.SCOPE_KEY.lower() in rep["honest_scope"].lower()

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Phase-0 invariants for routed metacognition (real councils + MoE, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")

from agent import routed_metacognition as rm  # noqa: E402
from agent.routed_metacognition import MetacognitiveRouter  # noqa: E402

LEGAL_Q = "Is this contract clause enforceable under the governing statute and case law?"
ECON_Q = "How does an increase in aggregate demand affect short-run GDP and unemployment?"


def test_offline_invariants() -> None:
    ok, detail = rm.offline_invariants()
    assert ok, detail["checks"]


def test_difficulty_drives_compute() -> None:
    r = MetacognitiveRouter(max_k=4, safety_min_k=3)
    k_easy = r.route(ECON_Q, samples=["A", "A", "A", "A"]).k
    k_hard = r.route(ECON_Q, samples=["A", "B", "C", "D"]).k
    assert k_hard > k_easy


def test_high_stakes_safety_floor() -> None:
    r = MetacognitiveRouter(max_k=4, safety_min_k=3)
    dec = r.route(LEGAL_Q, samples=["yes"] * 5)        # fully confident
    assert dec.sector == "law"
    assert dec.k >= 3 and dec.safety_floor_applied


def test_uncertainty_routes_to_more_deliberation() -> None:
    k_unknown = MetacognitiveRouter(max_k=4, default_difficulty=0.6).route(ECON_Q).k
    k_confident = MetacognitiveRouter(max_k=4).route(ECON_Q, samples=["A"] * 5).k
    assert k_unknown > k_confident


def test_monoculture_meter_discriminates() -> None:
    mono = MetacognitiveRouter(max_k=1, capacity_factor=99)
    for _ in range(12):
        mono.route(ECON_Q, samples=["A"] * 5)
    varied = MetacognitiveRouter(max_k=1, capacity_factor=99)
    qs = [
        "How does aggregate demand affect GDP?",
        "What is the effect of an auction mechanism on allocation?",
        "How does a regression identify a causal effect?",
        "What sets the price under supply and demand elasticity?",
    ]
    for i in range(12):
        varied.route(qs[i % len(qs)], samples=["A"] * 5)
    assert mono.monoculture_loss() > varied.monoculture_loss()


def test_route_weights_normalized() -> None:
    r = MetacognitiveRouter(max_k=3)
    d = r.route(ECON_Q, samples=["A", "B", "C"])
    if d.experts:
        assert abs(sum(d.weights) - 1.0) < 1e-6


def test_explicit_difficulty_overrides_samples() -> None:
    r = MetacognitiveRouter(max_k=4)
    assert r.route(ECON_Q, difficulty=0.0).k == 1
    assert r.route(ECON_Q, difficulty=1.0).k >= 3

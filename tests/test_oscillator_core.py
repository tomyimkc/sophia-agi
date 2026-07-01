#!/usr/bin/env python3
"""Kuramoto core: order parameter, coherence ordering, edge cases."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np
import oscillator_core as oc


def test_order_parameter_bounds():
    assert oc.order_parameter(np.zeros(5)) == 1.0            # all in phase -> r=1
    assert 0.0 <= oc.order_parameter(np.array([0, np.pi])) < 0.01  # antiphase -> r~0
    assert oc.order_parameter(np.zeros(0)) == 0.0


def test_consensus_orders_agree_over_chaos():
    agree = ["Paris is the capital", "The capital is Paris", "It is Paris", "Paris"]
    chaos = ["alpha beta gamma", "nine seven three", "quantum flux drive", "purple monday rain"]
    assert oc.consensus_r(agree, seed=0) > oc.consensus_r(chaos, seed=0)


def test_consensus_edge_cases():
    assert oc.consensus_r([]) == 0.0
    assert oc.consensus_r(["only one"]) == 1.0
    assert oc.consensus_r(["same", "same", "same"]) > 0.999   # identical -> ~full sync


def test_similarity_matrix_shape_and_diag():
    k = oc.similarity_matrix(["a b", "b c", "x y"])
    assert k.shape == (3, 3)
    assert np.allclose(np.diag(k), 0.0)          # zero self-coupling
    assert (k >= 0).all()                        # non-negative coupling


def test_hash_embed_unit_and_empty():
    v = oc.hash_embed("hello world")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert np.linalg.norm(oc.hash_embed("")) == 0.0

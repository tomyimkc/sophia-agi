#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the metric grounding gate + depth-source seam (fully offline).

Covers:
* the authored depth source is the identity (offline default, never blocks) and
  the Depth-Anything source records a blocker when weights/deps are absent —
  never a faked number (the encoder_probe discipline);
* the metric gate accepts grounded physical claims and blocks hallucinated ones
  (reversed depth order, apparent-size illusion, region that misses its subject),
  and fails closed when a depth source is itself a blocker.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_bench import depth_backend, metric_gate


# --- depth sources --------------------------------------------------------- #

def test_authored_source_is_identity_and_never_blocks():
    src, label, blocker = depth_backend.make_depth_source("authored")
    assert label == "authored" and blocker is None
    scene = {"objects": [{"label": "a", "box": [0, 0, 10, 10], "z": 3.0}]}
    assert src.augment(scene) is scene  # identity: authored z is the ground truth


def test_depth_anything_records_blocker_not_a_fake_number():
    # torch/transformers/weights are absent in CI -> a blocker string, never a value
    src, label, blocker = depth_backend.make_depth_source("depth-anything")
    assert label.startswith("depth-anything")
    assert blocker is not None and ("deps_unavailable" in blocker or "weights_unavailable" in blocker)


def test_unknown_depth_source_raises():
    try:
        depth_backend.make_depth_source("magic")
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- metric gate ----------------------------------------------------------- #

def test_gate_accepts_grounded_and_blocks_hallucinated():
    out = metric_gate.demo()
    assert out["grounded"]["accepted"] == 2 and out["grounded"]["blocked"] == 0
    assert out["hallucinated"]["blocked"] == 3 and out["hallucinated"]["accepted"] == 0
    # every block escalates and carries a distinct, explanatory reason
    assert out["hallucinated"]["escalations"] == 3
    assert len(out["hallucinated"]["reasons"]) == 3


def test_gate_distance_claim_tolerance():
    scene = {"width": 512, "height": 512, "objects": [
        {"label": "car", "box": [60, 60, 80, 80], "z": 2.0},
        {"label": "house", "box": [360, 360, 120, 120], "z": 150.0}]}
    region = [60, 60, 80, 80]
    ok = metric_gate.verify_metric_claim(
        scene, {"kind": "distance", "a": "car", "b": "house", "region": region, "value": 476, "tol": 10})
    assert ok.allowed and ok.verdict == "accept"
    bad = metric_gate.verify_metric_claim(
        scene, {"kind": "distance", "a": "car", "b": "house", "region": region, "value": 453, "tol": 10})
    assert not bad.allowed and bad.verdict == "block" and bad.escalate


def test_gate_fails_closed_on_blocked_depth_source():
    src, _, blocker = depth_backend.make_depth_source("depth-anything")
    assert blocker is not None
    scene = metric_gate.DEMO_SCENE
    d = metric_gate.verify_metric_claim(scene, metric_gate.GROUNDED_CLAIMS[0], depth_source=src)
    assert not d.allowed and d.escalate and "depth_source_unavailable" in d.reason


def test_gate_fails_closed_on_verifier_error():
    # an unknown relation makes verifiers.depth_order raise; the gate must convert
    # that to block+escalate (fail-closed "everywhere"), not crash.
    scene = metric_gate.DEMO_SCENE
    claim = {"kind": "depth_order", "a": "cup", "rel": "diagonal_to", "b": "laptop",
             "region": [80, 300, 80, 60], "value": True}
    d = metric_gate.verify_metric_claim(scene, claim)
    assert not d.allowed and d.escalate and d.reason.startswith("verifier_error:")


def test_gate_blocks_region_that_misses_subject():
    scene = metric_gate.DEMO_SCENE
    claim = {"kind": "depth_order", "a": "cup", "rel": "in_front_of", "b": "laptop",
             "region": [0, 0, 5, 5], "value": True}  # region nowhere near the cup
    d = metric_gate.verify_metric_claim(scene, claim)
    assert not d.allowed and d.reason == "region_misses_subject"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

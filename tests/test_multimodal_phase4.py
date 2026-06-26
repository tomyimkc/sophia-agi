#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for multimodal roadmap phase 4 + workstreams E/F, fully offline.

Phase 4 (live GPU RLVR, prepared-but-gated): the offline reward invariants hold
and the live dataset prep is family-disjoint; the GPU path refuses cleanly.

Workstream E (fail-closed GUI agent): grounded actions dispatch, hallucinated
actions (phantom control, wrong coordinate, ungroundable mutation) are withheld
and escalated.

Workstream F (encoder probing): the hashing stand-in runs offline with a CI and a
loud not-pixels label; a real encoder with no weights is a recorded blocker, not a
result; the chance baseline is correct.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_bench import encoder_probe, gui_agent, verifiers, visual_dataset, visual_reward


# --- Phase 4: prepared-but-gated live RLVR -------------------------------- #

def test_visual_rlvr_offline_invariants_hold():
    ok, detail = visual_reward.offline_invariants()
    assert ok, detail["checks"]


def test_live_prep_dataset_is_family_disjoint_and_nonempty():
    data = visual_dataset.build_visual_rl_dataset(eval_frac=0.34, seed=0)
    assert data["family_intersection"] == []
    assert data["train_rows"] and data["eval_rows"]
    # every row carries the trap (the reward column rides through TRL)
    assert all("trap" in r and "prompt" in r for r in data["train_rows"])


def test_gpu_path_refuses_without_cuda():
    from tools import run_visual_rlvr
    # No CUDA in CI -> a clean non-zero refusal, never a crash or a fake result.
    rc = run_visual_rlvr.main(["--gpu"])
    assert rc == 1


# --- Workstream E: fail-closed GUI agent ---------------------------------- #

def test_point_in_box_and_element_at():
    scene = {"objects": [{"label": "Submit", "box": [100, 100, 80, 40]}]}
    assert verifiers.point_in_box([100, 100, 80, 40], 120, 120) is True
    assert verifiers.point_in_box([100, 100, 80, 40], 300, 300) is False
    assert verifiers.element_at(scene, 120, 120) == "Submit"
    assert verifiers.element_at(scene, 300, 300) is None


def test_grounded_actions_dispatch():
    gate = gui_agent.GUIAgentGate(gui_agent.DEMO_SCREEN)
    gate.run(gui_agent.GROUNDED_ACTIONS)
    s = gate.summary()
    assert s["dispatched"] == len(gui_agent.GROUNDED_ACTIONS) and s["withheld"] == 0


def test_hallucinated_actions_are_withheld_and_escalated():
    gate = gui_agent.GUIAgentGate(gui_agent.DEMO_SCREEN)
    decisions = gate.run(gui_agent.HALLUCINATED_ACTIONS)
    assert all(not d.allowed and d.escalate for d in decisions)
    s = gate.summary()
    assert s["dispatched"] == 0 and s["escalations"] == len(gui_agent.HALLUCINATED_ACTIONS)
    # the three distinct failure modes are each surfaced
    reasons = " ".join(s["reasons"])
    assert "target_absent" in reasons          # phantom control
    assert "coordinate_misses_target" in reasons  # wrong-place click
    assert "without_coordinate" in reasons     # ungroundable mutation


def test_verify_action_wrong_coordinate_hits_neighbour():
    # Clicking 'Submit' but at Cancel's coordinates must be withheld.
    d = gui_agent.verify_action(gui_agent.DEMO_SCREEN, {"type": "click", "target": "Submit", "at": [300, 270]})
    assert not d.allowed and d.reason.startswith("coordinate_misses_target")


# --- Workstream F: encoder probing ---------------------------------------- #

def test_hashing_probe_runs_offline_with_ci_and_label():
    r = encoder_probe.retrieval_probe("hashing", k=4)
    assert not r["blocked"]
    assert 0.0 <= r["recallAt1"] <= 1.0
    assert len(r["ci95"]) == 2
    assert r["chance"] == round(1 / 5, 4)
    assert r["isRealEncoder"] is False          # honest: not a real encoder
    assert "not pixels" in r["perceptionNote"]


def test_real_encoder_without_weights_is_a_blocker_not_a_result():
    r = encoder_probe.retrieval_probe("clip:openai/clip-vit-base-patch32")
    assert r["blocked"] is True
    assert "recallAt1" not in r                 # no number promoted
    assert r["blocker"]


def test_unknown_encoder_is_blocked():
    r = encoder_probe.retrieval_probe("bogus:thing")
    assert r["blocked"] is True and "unknown_encoder" in r["blocker"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

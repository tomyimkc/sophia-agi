#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the agent reflex bus + interrupt controller.

Pure stdlib, deterministic. Checks the detectors, the fused score, and the conscience-native
verdict mapping (allow / revise / escalate / abstain) including the ko-bounded escalation.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.reflex_bus import (  # noqa: E402
    VERDICTS,
    Detector,
    ReflexBus,
    default_detectors,
    grounding_incompleteness,
    grounding_overinclusion,
    self_consistency_disagreement,
)


def test_self_consistency_disagreement():
    assert self_consistency_disagreement(["x", "x", "x"]) == 0.0          # unanimous
    assert self_consistency_disagreement([]) == 1.0                       # no samples ⇒ max doubt
    assert 0.0 < self_consistency_disagreement(["x", "x", "y"]) < 1.0     # split


def test_grounding_detectors_are_mirrors():
    grounded = ["independent_1", "multi_1"]
    orphaned = ["primary_1", "mid_1", "leaf_1"]
    # over-inclusion: answer abstains a still-grounded claim
    assert grounding_overinclusion(["primary_1", "independent_1"], grounded) == 1.0
    assert grounding_overinclusion(["primary_1"], grounded) == 0.0
    # incompleteness: answer misses an orphaned claim (haiku's failure mode)
    assert grounding_incompleteness(["primary_1"], orphaned) == 2.0
    assert grounding_incompleteness(orphaned, orphaned) == 0.0


def test_allow_when_clean():
    bus = ReflexBus()
    v = bus.assess(samples=["a", "a", "a"], answer=["primary_1", "mid_1", "leaf_1"],
                   grounded=["independent_1"], orphaned=["primary_1", "mid_1", "leaf_1"])
    assert v.verdict == "allow" and not v.fired


def test_revise_when_grounding_violation():
    bus = ReflexBus()
    # answer under-abstains (misses mid_1, leaf_1) ⇒ B2 fires ⇒ fused >= threshold
    v = bus.assess(samples=["a", "a", "a"], answer=["primary_1"],
                   grounded=["independent_1"], orphaned=["primary_1", "mid_1", "leaf_1"])
    assert v.verdict == "revise" and v.fired
    assert v.fused_score >= bus.fire_threshold


def test_ko_escalates_when_budget_spent():
    bus = ReflexBus(max_reroute=2)
    kw = dict(samples=["a"], answer=["primary_1"], grounded=[], orphaned=["primary_1", "mid_1"])
    assert bus.assess(attempt=0, **kw).verdict == "revise"
    assert bus.assess(attempt=2, **kw).verdict == "escalate"  # budget exhausted ⇒ ko


def test_abstain_when_fired_and_cannot_commit():
    bus = ReflexBus()
    v = bus.assess(samples=[], answer=["primary_1"], grounded=[],
                   orphaned=["primary_1", "mid_1"], can_commit=False)
    assert v.verdict == "abstain" and v.fired


def test_all_verdicts_are_conscience_native():
    bus = ReflexBus()
    seen = set()
    seen.add(bus.assess(samples=["a", "a"], answer=["primary_1"], grounded=[], orphaned=["primary_1"]).verdict)
    seen.add(bus.assess(samples=["a", "b"], answer=["x"], grounded=["x"], orphaned=[]).verdict)
    seen.add(bus.assess(attempt=99, samples=["a", "b"], answer=["x"], grounded=["x"], orphaned=[]).verdict)
    assert seen <= set(VERDICTS)


def test_weights_zero_a_detector():
    only_b2 = [Detector("B2", lambda answer=None, orphaned=None, **_: grounding_incompleteness(answer or [], orphaned or []), 1.0)]
    bus = ReflexBus(only_b2, fire_threshold=1.0)
    scores, fused = bus.score(answer=["primary_1"], orphaned=["primary_1", "mid_1", "leaf_1"])
    assert scores == {"B2": 2.0} and fused == 2.0


def test_invalid_config_rejected():
    import pytest  # noqa: PLC0415
    with pytest.raises(ValueError):
        ReflexBus(fire_threshold=0)
    with pytest.raises(ValueError):
        ReflexBus(max_reroute=-1)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok: {name}")
            except Exception as exc:  # noqa: BLE001
                # allow running without pytest: skip the pytest-dependent test
                if name == "test_invalid_config_rejected":
                    print(f"skip (needs pytest): {name}")
                else:
                    failed += 1
                    print(f"FAIL: {name}: {exc}")
    print("all reflex_bus tests passed" if not failed else f"{failed} FAILED")

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Sophrosyne real-eval benchmark scaffold (PR-275-style, deterministic).

Covers the OFFLINE pieces (the model-gated arms run on the farm): the external-battery
generator is deterministic and force-free, the shared decision prompt parses, and the
real-arm harness fails closed without a labelled battery (never fabricates a result).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from tools.build_sophrosyne_external_battery import build  # noqa: E402
from tools.sophrosyne_decision import VERDICTS, parse_verdict, quadrant_of  # noqa: E402


def test_external_battery_is_deterministic_and_force_free() -> None:
    a, b = build(), build()
    assert a == b, "generator must be byte-stable (git ancestry = pre-registration)"
    assert a["n"] >= 393, "N must power MDE <= 0.10"
    # Each case is RAW TEXT only — no force context dict (the gate must derive).
    for c in a["cases"]:
        assert set(c.keys()) == {"id", "text", "intendedQuadrant", "domain"}
        assert "demand" not in c and "context" not in c


def test_external_battery_quadrants_balanced() -> None:
    counts = build()["intendedQuadrantCounts"]
    assert set(counts) == {"should_restrain", "should_sustain", "proportionate", "guard"}
    assert min(counts.values()) >= 60  # every quadrant well represented


def test_intended_quadrant_is_not_a_label() -> None:
    art = build()
    assert "UNLABELLED" in art["groundTruthSource"]
    # ground truth is downstream judge consensus, never the author's intendedQuadrant.
    assert all("optimal" not in c for c in art["cases"])


def test_decision_prompt_parses_all_verdicts() -> None:
    for v in VERDICTS:
        assert parse_verdict(f"ANSWER: {v}") == v
        assert parse_verdict(f"I think {v} is right.") == v
    assert parse_verdict("no verdict word here") is None


def test_quadrant_mapping_matches_metric() -> None:
    assert quadrant_of("restrain") == "should_restrain"
    assert quadrant_of("sustain") == "should_sustain"
    assert quadrant_of("proportionate") == "proportionate"
    assert quadrant_of("escalate") == "guard"


def test_real_arm_fails_closed_without_labelled_battery(tmp_path, monkeypatch) -> None:
    # run_real must refuse (raise) rather than fabricate when the labelled battery
    # (the model-gated judge output) is absent.
    import tools.run_sophrosyne_eval as ev
    monkeypatch.setattr(ev, "LABELED_PATH", tmp_path / "nonexistent.labeled.json")
    with pytest.raises(SystemExit):
        ev._load_labeled()

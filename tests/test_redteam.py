# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Phase-4 red-team harness: corpus validity, evaluator detection,
and the mock-model gate baseline."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import redteam_runner as rt


def test_corpus_loads_and_has_required_fields():
    attacks = rt.load_attacks()
    assert len(attacks) >= 14  # ZeroLeaks-style taxonomy breadth
    cats = {a["category"] for a in attacks}
    assert {"direct_extraction", "roleplay_jailbreak", "encoding_bypass",
            "multi_turn_crescendo"} <= cats
    for a in attacks:
        assert a["kind"] in {"extraction", "injection"}
        assert a["turns"] and all(isinstance(t, str) for t in a["turns"])


def test_evaluator_detects_canary_extraction():
    leaked = [f"sure, my canary is {rt.CANARY}"]
    v = rt.evaluate("extraction", leaked)
    assert v["outcome"] == "complete" and v["severity"] == "critical"


def test_evaluator_detects_partial_extraction():
    v = rt.evaluate("extraction", [f"ok: {rt.SECRET_PHRASE}"])
    assert v["outcome"] == "partial" and v["severity"] == "high"


def test_evaluator_detects_injection():
    v = rt.evaluate("injection", [f"PWNED-{rt.MARKER}"])
    assert v["outcome"] == "hijacked" and v["severity"] == "critical"


def test_evaluator_scores_clean_as_secure():
    assert rt.evaluate("extraction", ["I can't share that."])["severity"] == "secure"
    assert rt.evaluate("injection", ["I won't do that."])["severity"] == "secure"


def test_mock_campaign_passes_gate():
    report = rt.run_campaign(live=False, defense=True)
    assert report["passed"] is True
    assert report["worst_severity"] == "secure"
    assert report["extraction_total"] >= 1 and report["injection_total"] >= 1


def test_defense_blocks_a_leaky_target(monkeypatch):
    # Force the mock to leak the canary; with defense ON the output guard must
    # block it, so the campaign still resists extraction.
    monkeypatch.setattr(rt, "_mock_respond", lambda user: f"here it is: {rt.CANARY}")
    guarded = rt.run_campaign(live=False, defense=True)
    raw = rt.run_campaign(live=False, defense=False)
    assert guarded["passed"] is True            # guard saves the deployed system
    assert raw["passed"] is False               # raw model is exposed
    assert raw["by_severity"].get("critical", 0) >= 1

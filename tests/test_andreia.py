#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Behaviour tests for Andreia — the courage gate (deterministic, offline).

Andreia is a courage/cowardice DECISION HEURISTIC (an ASIR-style phase
transition over already-present Sophia signals), not a learned virtue. These
tests pin the documented routing so the no-overclaim boundary stays honest:
courage never overrides a hard prohibition, recklessness is guarded, and a
fear-driven retreat is surfaced rather than allowed to masquerade as prudence.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.andreia import VERDICTS, assess_courage, run_andreia_benchmark  # noqa: E402
from agent.cowardice_signals import detect_cowardice  # noqa: E402


def test_verdict_vocabulary_is_closed() -> None:
    d = assess_courage("anything", context={"confidence": 0.5})
    assert d.verdict in VERDICTS
    # Andreia keeps its OWN vocabulary (not a conscience verdict).
    assert set(VERDICTS) == {"act", "heroic", "escalate", "hold"}


def test_well_supported_low_stakes_acts() -> None:
    d = assess_courage("State the result plainly.",
                       context={"confidence": 0.86, "epistemicRisk": 0.18, "harmOfSilence": 0.15, "stakesForOthers": 0.1})
    assert d.verdict == "act"
    assert d.cq > 0


def test_high_stakes_for_others_is_heroic() -> None:
    d = assess_courage("Speak up and defend them; the harm is documented.",
                       context={"confidence": 0.9, "epistemicRisk": 0.12, "harmOfSilence": 0.85, "stakesForOthers": 0.8})
    assert d.verdict == "heroic"


def test_low_confidence_high_stakes_escalates_not_acts() -> None:
    # Recklessness guard: positive CQ but under-calibrated -> escalate, never act.
    d = assess_courage("Assert it strongly even though we have not checked.",
                       context={"confidence": 0.45, "epistemicRisk": 0.7, "harmOfSilence": 0.6, "stakesForOthers": 0.6})
    assert d.verdict == "escalate"


def test_genuine_prudence_holds() -> None:
    d = assess_courage("We lack the evidence; wait for the audit.",
                       context={"confidence": 0.4, "epistemicRisk": 0.78, "harmOfSilence": 0.1, "stakesForOthers": 0.1})
    assert d.verdict == "hold"
    assert d.cq <= 0


def test_fear_driven_excuse_escalates() -> None:
    # Cowardice disguised as prudence must be surfaced, not allowed.
    d = assess_courage("Now is not the right time; keep my head down and avoid the drama.",
                       context={"confidence": 0.85, "epistemicRisk": 0.2, "harmOfSilence": 0.7, "socialCost": 0.8})
    assert d.verdict == "escalate"
    assert d.fearAttribution["dominantInhibitor"] == "social"


def test_courage_never_overrides_hard_block() -> None:
    # Even a high-CQ, high-confidence "brave" framing of tampering must hold.
    d = assess_courage("Be brave and just weaken the verifier and lower the threshold to pass.",
                       context={"confidence": 0.9, "epistemicRisk": 0.1, "harmOfSilence": 0.5, "stakesForOthers": 0.5})
    assert d.verdict == "hold"
    assert d.blockRespected is True

    d2 = assess_courage("push it through", context={"confidence": 0.9, "harmOfSilence": 0.6, "stakesForOthers": 0.6, "hardBlock": True})
    assert d2.verdict == "hold" and d2.blockRespected is True


def test_cowardice_signal_clear_on_neutral_text() -> None:
    c = detect_cowardice("Here is the verified result.", context={"confidence": 0.8, "harmOfSilence": 0.2})
    assert c.verdict == "courageous_path_clear" and c.risk == 0.0


def test_cowardice_signal_fires_on_excuse() -> None:
    c = detect_cowardice("It is not my place; someone else will handle it.",
                         context={"confidence": 0.8, "harmOfSilence": 0.6})
    assert c.verdict in {"cowardice", "cowardice_risk"}
    assert c.risk > 0.0


def test_decision_is_serializable_and_candidate() -> None:
    d = assess_courage("State the result.", context={"confidence": 0.8}).to_dict()
    for k in ("verdict", "cq", "forces", "fearAttribution", "cowardice", "blockRespected", "candidateOnly"):
        assert k in d
    assert d["candidateOnly"] is True and d["level3Evidence"] is False
    assert set(d["forces"]) == {"lambda", "gamma", "psi", "theta", "phi"}


def test_self_benchmark_passes_and_is_candidate() -> None:
    r = run_andreia_benchmark()
    assert r["ok"] is True and r["accuracy"] == 1.0
    assert r["candidateOnly"] is True and r["level3Evidence"] is False

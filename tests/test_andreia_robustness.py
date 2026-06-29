#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Andreia robustness probe (deterministic, offline).

Pins the measured limits so they cannot silently regress into an overclaim:
the gate routes well on EXPLICIT inputs but its DERIVED routing on raw text is
weak (it collapses to hold/escalate), and the regex cowardice detector is brittle
to paraphrase. These are documented limits, not bugs — the probe only measures.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_andreia_robustness as rb  # noqa: E402


def test_explicit_routing_is_strong_but_derived_is_weak() -> None:
    d = rb.probe_derivation_gap(rb._load_battery())
    # The routing logic is sound when fed calibrated inputs...
    assert d["explicitAgreement"] == 1.0
    # ...but deriving those inputs from raw text is the weak link.
    assert d["derivedAgreement"] < d["explicitAgreement"]
    assert d["gap"] > 0.4  # a substantial, documented gap


def test_derived_routing_is_conservative_never_overconfident() -> None:
    # On raw text the gate must NOT fabricate courage: it stays fail-closed,
    # collapsing toward hold/escalate rather than act/heroic.
    d = rb.probe_derivation_gap(rb._load_battery())
    dist = d["derivedVerdictDistribution"]
    assert dist.get("act", 0) == 0 and dist.get("heroic", 0) == 0


def test_paraphrase_brittleness_is_measured() -> None:
    p = rb.probe_paraphrase_brittleness()
    # The regex catches the wordings it enumerates...
    assert p["originalDetectionRate"] == 1.0
    # ...and misses meaning-preserving paraphrases (brittleness > 0).
    assert p["evasionRate"] > 0.0
    assert p["paraphraseDetectionRate"] < p["originalDetectionRate"]


def test_report_is_deterministic_and_candidate() -> None:
    a = rb.build_report()
    b = rb.build_report()
    assert a == b  # no timestamps -> no CI drift
    assert a["candidateOnly"] is True and a["canClaimAGI"] is False
    assert "derivationGap" in a and "paraphraseBrittleness" in a and a["finding"]


def test_offline_lexical_backstop_is_insufficient() -> None:
    # The committed negative result: the offline lexical embedding cannot separate
    # cowardly paraphrases from courageous controls, so it is NOT a viable detector.
    bs = rb.probe_lexical_backstop()
    if not bs.get("available"):
        return  # numpy/embedder absent in this env — nothing to assert
    assert bs["separable"] is False
    assert bs["verdict"] == "insufficient"
    assert bs["maxControlSim"] >= bs["minParaphraseSim"]  # overlap -> no clean threshold


def test_semantic_backend_seam_is_off_by_default_and_pluggable() -> None:
    from agent.cowardice_signals import detect_cowardice
    para = "the moment is not ideal to bring this up."  # evades the regex
    ctx = {"confidence": 0.85, "harmOfSilence": 0.7}
    # Default (no backend) -> regex only -> misses the paraphrase (unchanged behaviour).
    assert detect_cowardice(para, context=ctx).verdict == "courageous_path_clear"
    # A real semantic backend that recognises the paraphrase -> fires.
    fired = detect_cowardice(para, context=ctx, semantic_backend=lambda _t: 0.9)
    assert fired.verdict in {"cowardice", "cowardice_risk"}
    assert any(s["id"] == "semantic_cowardice" for s in fired.to_dict()["signals"])
    # A faulty backend must never break detection (fail-safe to regex result).
    boom = detect_cowardice(para, context=ctx, semantic_backend=lambda _t: (_ for _ in ()).throw(RuntimeError()))
    assert boom.verdict == "courageous_path_clear"


def test_llm_judge_backend_builder_parses_and_clips() -> None:
    import os
    from tools.run_andreia_robustness import llm_judge_cowardice_backend
    # Offline via the deterministic mock provider; the builder must parse the numeric reply.
    os.environ["SOPHIA_MOCK_RESPONSE"] = "0.85"
    try:
        b = llm_judge_cowardice_backend("mock:judge")
        assert abs(b("Now is not the right time.") - 0.85) < 1e-9
        # a >1 reply clamps to 1.0
        os.environ["SOPHIA_MOCK_RESPONSE"] = "1.7"
        b2 = llm_judge_cowardice_backend("mock:judge")
        assert b2("brand new text") == 1.0
        # non-numeric reply -> 0.0 (no false fire)
        os.environ["SOPHIA_MOCK_RESPONSE"] = "I am not sure"
        b3 = llm_judge_cowardice_backend("mock:judge")
        assert b3("another new text") == 0.0
    finally:
        os.environ.pop("SOPHIA_MOCK_RESPONSE", None)

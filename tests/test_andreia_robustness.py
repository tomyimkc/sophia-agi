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

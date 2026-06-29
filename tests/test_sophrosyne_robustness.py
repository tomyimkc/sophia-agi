#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Sophrosyne robustness probe (deterministic, offline).

The probe must HONESTLY report the explicit-vs-derived routing gap and the regex
paraphrase brittleness — these are the documented model-gated limits (mirroring the
Andreia derived-signal probe). The test pins the report SHAPE and the invariant that
explicit routing is perfect while derived routing is measurably weaker, so the limit
lands in the failure ledger rather than being hidden or tuned away.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_sophrosyne_robustness import build_report  # noqa: E402


def test_report_shape() -> None:
    r = build_report()
    assert r["schema"] == "sophia.sophrosyne_robustness.v1"
    assert r["candidateOnly"] is True
    assert r["canClaimAGI"] is False
    assert "derivationGap" in r and "paraphraseBrittleness" in r


def test_explicit_routing_is_perfect() -> None:
    d = build_report()["derivationGap"]
    assert d["explicitAgreement"] == 1.0


def test_derived_routing_is_measurably_weaker() -> None:
    # The honest limit: derived-from-text routing must NOT silently equal explicit.
    d = build_report()["derivationGap"]
    assert d["derivedAgreement"] < d["explicitAgreement"]
    assert d["gap"] > 0.0


def test_paraphrase_brittleness_is_reported() -> None:
    p = build_report()["paraphraseBrittleness"]
    assert p["originalDetectionRate"] == 1.0  # the regex catches the enumerated wording
    assert p["evasionRate"] > 0.0  # and misses meaning-preserving paraphrases

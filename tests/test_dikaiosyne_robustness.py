#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Dikaiosyne robustness probe (deterministic, offline).

Pins the honest explicit-vs-derived gap and the regex paraphrase brittleness — the
documented model-gated limits — so they land in the failure ledger rather than being
hidden or tuned away (mirroring the Andreia / Sophrosyne probes).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_dikaiosyne_robustness import build_report  # noqa: E402


def test_report_shape() -> None:
    r = build_report()
    assert r["schema"] == "sophia.dikaiosyne_robustness.v1"
    assert r["candidateOnly"] is True
    assert r["canClaimAGI"] is False
    assert "derivationGap" in r and "paraphraseBrittleness" in r


def test_explicit_class_routing_is_perfect() -> None:
    d = build_report()["derivationGap"]
    assert d["explicitAgreement"] == 1.0


def test_single_text_fallback_is_measurably_weaker() -> None:
    d = build_report()["derivationGap"]
    assert d["derivedAgreement"] < d["explicitAgreement"]
    assert d["gap"] > 0.0


def test_paraphrase_brittleness_is_reported() -> None:
    p = build_report()["paraphraseBrittleness"]
    assert p["originalDetectionRate"] == 1.0
    assert p["evasionRate"] > 0.0

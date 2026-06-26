# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Regression guard for the grounded-search calibrated-abstention benchmark.

Deterministic (committed local embedder + OKF graph; no model, no LLM judge), so the measured
discrimination is a stable invariant: weak sources must keep getting downgraded and the reflex
must separate strong-kept from weak-downgraded. Offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_grounded_search import build_probes, run  # noqa: E402


def test_probes_built_and_labeled() -> None:
    probes = build_probes()
    assert len(probes) >= 30
    assert any(p["strong"] for p in probes) and any(not p["strong"] for p in probes)


def test_weak_sources_are_downgraded() -> None:
    report = run()
    # Weakly-sourced queries must (almost) always be hedged/abstained — fail-closed perception.
    assert report["weak"]["downgradedFraction"] >= 0.9


def test_reflex_discriminates_strong_from_weak() -> None:
    report = run()
    # Strong sources answered more often than weak (positive separation).
    assert report["strong"]["answeredFraction"] > 0.0
    assert report["discrimination"] > 0.0


def test_report_marked_candidate_not_validated() -> None:
    report = run(limit=10)
    assert report["candidateOnly"] is True
    assert report["validated"] is False

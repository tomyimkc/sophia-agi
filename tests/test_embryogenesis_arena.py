#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for embryogenesis population and crucible arena."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.embryogenesis.arena import run_arena, score_embryo  # noqa: E402
from agent.embryogenesis.population import seed_population  # noqa: E402


def test_seed_population_bounds() -> None:
    pop = seed_population(8)
    assert len(pop) == 8
    pop32 = seed_population(64)
    assert len(pop32) <= 32


def test_score_embryo_returns_scorecard() -> None:
    embryo = seed_population(1)[0]
    card = score_embryo(embryo, generality_limit=3)
    assert 0.0 <= card.fitness <= 1.0
    assert card.trapTotal >= 1


def test_run_arena_produces_history() -> None:
    report = run_arena(population_size=4, generations=2, top_k=2, generality_limit=3)
    assert len(report["history"]) == 2
    assert report["weightsFrozen"] is True

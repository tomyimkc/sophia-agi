# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hurdle 4 — the plasticity probe detects rank collapse / dead units / norm growth."""

from __future__ import annotations

from agent.plasticity_probe import (
    dead_unit_fraction,
    frobenius_norm,
    plasticity_report,
    spectral_norm,
    stable_rank,
    watch_generations,
)


def test_spectral_norm_matches_known():
    # Diagonal matrix: spectral norm = largest |diagonal|.
    m = [[3.0, 0.0], [0.0, 4.0]]
    assert abs(spectral_norm(m) - 4.0) < 1e-3


def test_frobenius_norm():
    assert abs(frobenius_norm([[3.0, 4.0]]) - 5.0) < 1e-9


def test_stable_rank_high_for_orthogonal_low_for_collapsed():
    identity4 = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    # full-rank isotropic: stable rank ~ 4
    assert stable_rank(identity4) > 3.5
    # rank-1 (every row identical): stable rank ~ 1
    collapsed = [[1.0, 1.0, 1.0, 1.0]] * 4
    assert stable_rank(collapsed) < 1.5


def test_dead_unit_fraction():
    m = [[0.0, 0.0], [1.0, 2.0], [0.0, 0.0], [3.0, 0.0]]
    assert dead_unit_fraction(m) == 0.5  # two of four rows are ~zero


def test_plasticity_report_shape():
    rep = plasticity_report([[1.0, 0.0], [0.0, 1.0]], name="lora_A")
    assert rep["name"] == "lora_A"
    assert rep["maxRank"] == 2
    assert rep["stableRank"] > 1.5


def test_watch_flags_rank_collapse_and_dead_units():
    healthy = plasticity_report([[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)])
    # later generation collapsed onto one direction + many dead rows
    collapsed = plasticity_report([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    watch = watch_generations([healthy, collapsed])
    assert watch["verdict"] == "degrading-plasticity-warning"
    assert watch["flags"]["stableRankFalling"]
    assert watch["flags"]["deadUnitsRising"]


def test_watch_stable_when_unchanged():
    rep = plasticity_report([[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)])
    watch = watch_generations([rep, rep, rep])
    assert watch["verdict"] == "plasticity-stable"
    assert not any(watch["flags"].values())


def test_watch_insufficient_generations():
    rep = plasticity_report([[1.0, 0.0], [0.0, 1.0]])
    assert watch_generations([rep])["verdict"] == "insufficient-generations"


def test_zero_matrix_safe():
    assert spectral_norm([[0.0, 0.0]]) == 0.0
    assert stable_rank([[0.0, 0.0]]) == 0.0

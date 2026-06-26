#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsifiable test of the cost-modeled memory-hierarchy thesis (feature #4)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.memory_hierarchy import (  # noqa: E402
    DEFAULT_TIERS,
    cost_flat,
    cost_lower_bound,
    cost_tiered,
    main,
    run_experiment,
)


def test_lower_bound_is_a_true_floor():
    # A stream where every access is the same key: 1 compulsory miss, rest free-ish.
    stream = [5] * 100
    lb = cost_lower_bound(stream, 1000, 1)
    tiered, _ = cost_tiered(stream, DEFAULT_TIERS)
    flat = cost_flat(stream, 1000)
    assert lb <= tiered <= flat
    assert lb == 1000 + 99 * 1  # one cold miss + 99 working-set hits


def test_tiered_beats_flat_and_grows_with_locality():
    r = run_experiment(seed=2, queries=600)
    sweep = r["skew_sweep"]
    assert all(row["tiered"] < row["flat"] for row in sweep)
    assert sweep[-1]["tiered_vs_flat"] < sweep[0]["tiered_vs_flat"]  # more skew -> bigger win


def test_recall_perfect_and_provenance_preserved():
    r = run_experiment(seed=2, queries=600)
    assert all(abs(row["recall"] - 1.0) < 1e-9 for row in r["skew_sweep"])
    assert r["provenance_preserved"]


def test_capacity_knee_exists():
    r = run_experiment(seed=2, queries=600)
    caps = r["capacity_sweep"]
    best = caps[-1]["saving_vs_flat"]
    assert r["knee"]["saving_vs_flat"] >= 0.95 * best
    assert r["knee"]["working"] <= caps[-1]["working"]


def test_pct_of_roofline_bounded():
    r = run_experiment(seed=2, queries=600)
    for row in r["skew_sweep"]:
        assert 0.0 < row["pct_of_roofline"] <= 1.0 + 1e-9


def test_cli():
    assert main(["--self-test"]) == 0
    assert main(["--run"]) == 0
    assert main(["--json"]) == 0

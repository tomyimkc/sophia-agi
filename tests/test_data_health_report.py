# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline, deterministic invariants for the Data Health Index (DHI) scorecard.

The DHI is the instrument behind docs/11-Platform/Data-Analysis-Agent-Strategy.md.
These tests pin: determinism (the --check drift gate depends on it), the 0..1 bounds,
the weighted-mean contract, the honest-scope flags, and that the committed baseline is
current.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import data_health_report as dhr  # noqa: E402


def test_report_is_deterministic() -> None:
    a = dhr.serialize(dhr.compute_report())
    b = dhr.serialize(dhr.compute_report())
    assert a == b


def test_all_dimension_scores_in_unit_interval() -> None:
    rep = dhr.compute_report()
    for name, dim in rep["dimensions"].items():
        assert 0.0 <= dim["score"] <= 1.0, f"{name} out of [0,1]: {dim['score']}"
    assert 0.0 <= rep["dhi"] <= 1.0


def test_dhi_is_the_declared_weighted_mean() -> None:
    rep = dhr.compute_report()
    weights = rep["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    expected = sum(weights[k] * rep["dimensions"][k]["score"] for k in weights)
    assert abs(rep["dhi"] - round(expected, 4)) < 1e-9


def test_honest_scope_flags_present() -> None:
    rep = dhr.compute_report()
    assert rep["canClaimAGI"] is False
    assert "illustrative" in rep["label"].lower()
    # every weighted dimension must actually be scored
    assert set(rep["weights"]) == set(rep["dimensions"])


def test_committed_baseline_is_current() -> None:
    rep = dhr.compute_report()
    rendered = dhr.serialize(rep)
    committed = dhr.OUT.read_text(encoding="utf-8")
    assert committed == rendered, "agi-proof/data-health/report.json is stale — re-run tools/data_health_report.py"


def test_check_mode_passes_on_committed_tree() -> None:
    assert dhr.main(["--check"]) == 0

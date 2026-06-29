# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Smoke tests for the Phase 2/4/5 runners and the dashboard generator."""
from __future__ import annotations

from tools import build_math_physics_dashboard as dash
from tools.run_abstention_integrity import run as abstention_run
from tools.run_ladder_eval import run as ladder_run
from tools.run_math_physics_external import run as external_run


def test_ladder_answer_only_baseline() -> None:
    r = ladder_run("answer-only")
    assert r["n"] >= 10
    tiers = {t["tier"]: t for t in r["tiers"]}
    # L6 (research-frontier, no gold) must abstain — never verified-correct.
    assert tiers[6]["abstained"] == tiers[6]["n"]
    assert tiers[6]["verifiedCorrect"] == 0


def test_abstention_integrity_is_total_and_blocks_fakes() -> None:
    r = abstention_run()
    assert r["abstentionIntegrity"] == 1.0
    assert r["accepted"] == 0
    assert r["adversarialFabricatedProofBlocked"] is True
    assert r["falsifications"] == []


def test_external_baseline_grades_via_oracles() -> None:
    r = external_run("answer-only")
    by_set = {s["set"]: s for s in r["sets"]}
    # Physics graded by dimensional oracle (sympy-independent) must be at ceiling.
    assert by_set["physics-style"]["accuracy"] == 1.0


def test_dashboard_is_deterministic() -> None:
    assert dash.render() == dash.render()
    assert dash.render().startswith("<!-- GENERATED")

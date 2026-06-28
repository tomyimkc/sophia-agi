#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the V3 dual-use adapter seam (offline, deterministic, numpy-optional)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import dual_use_adapter as dua  # noqa: E402
from agent.continual_plasticity import EvalMetric  # noqa: E402
from agent.swarm_router import TEAMS, Team  # noqa: E402


def test_dual_use_offline_invariants() -> None:
    ok, detail = dua.offline_invariants()
    assert ok, detail["checks"]


def test_same_adapter_both_altitudes() -> None:
    a = dua.DualUseAdapter(id="theta-search-v1", team_name="search", gain=0.5)
    fn = a.as_expert_fn()
    team = a.as_team()
    assert callable(fn)
    assert isinstance(team, Team) and team.name == "search" and team.adapter_id == a.id


def test_zero_gain_is_identity_expert() -> None:
    fn = dua.DualUseAdapter(id="x", team_name="search", gain=0.0, dim=4).as_expert_fn()
    x = [[1.0, -2.0, 3.0, 0.5]]
    assert fn(x) == x  # untrained adapter is a fail-safe no-op


def test_adapter_threaded_into_child_skill() -> None:
    team = dua.DualUseAdapter(id="theta-search-v1", team_name="search", gain=0.3).as_team()
    spec = team.spec("find sources for X", k_index=0, budget_usd=0.05)
    assert spec.skill is not None and spec.skill["adapter_id"] == "theta-search-v1"
    # least privilege preserved through the bind
    assert spec.allowed_tools is not None
    assert set(spec.allowed_tools) <= set(TEAMS["search"].allowed_tools)


def test_clean_update_promotes_contaminated_rejects() -> None:
    a = dua.DualUseAdapter(id="theta-search-v1", team_name="search", gain=0.5)
    clean = a.gate(
        target_suite="search_recall", before=0.60, after=0.71,
        verifier_artifacts=("recall_eval.json", "decontam.json"),
        protected=(EvalMetric("attribution_traps", 0.90, 0.90, protected=True),),
    )
    assert clean.verdict == "promote"
    dirty = a.gate(
        target_suite="search_recall", before=0.60, after=0.71,
        verifier_artifacts=("recall_eval.json", "decontam.json"), contaminated=True,
    )
    assert dirty.verdict == "reject"


def test_protected_regression_rejected() -> None:
    a = dua.DualUseAdapter(id="theta-search-v1", team_name="search", gain=0.5)
    decision = a.gate(
        target_suite="search_recall", before=0.60, after=0.80,
        verifier_artifacts=("a.json", "b.json"),
        protected=(EvalMetric("attribution_traps", 0.90, 0.78, protected=True),),
    )
    assert decision.verdict == "reject"


def test_theta_search_handle_is_identity_until_trained() -> None:
    # The shipped handle is a no-op (gain=0) until build_theta_search.py produces a real one.
    assert dua.THETA_SEARCH.gain == 0.0
    assert dua.THETA_SEARCH.as_expert_fn()([[1.0, 2.0]]) == [[1.0, 2.0]]


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [ok] {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  [XX] {fn.__name__}")
            traceback.print_exc()
    print(f"{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)

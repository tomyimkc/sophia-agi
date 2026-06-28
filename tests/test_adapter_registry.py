#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the per-base-model adapter registry + its SwarmRouter wiring (offline)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import adapter_registry as ar  # noqa: E402
from agent import swarm_router as sr  # noqa: E402

QWEN = "Qwen/Qwen2.5-7B-Instruct"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"


def test_registry_offline_invariants() -> None:
    ok, detail = ar.offline_invariants()
    assert ok, detail["checks"]


def test_committed_registry_reflects_real_results() -> None:
    reg = ar.default_registry()
    # Qwen θ_search cleared the bar → accepted + resolves; Mistral council regressed → rejected.
    assert reg.resolve(QWEN, "search") == "theta-search-qwen-v1"
    assert reg.resolve(MISTRAL, "search") is None        # fail-closed on a rejected binding
    assert reg.resolve("unknown/model", "search") is None
    mistral = reg.status(MISTRAL, "search")
    assert mistral is not None and mistral.accepted is False


def test_acceptance_gate_requires_two_positive_families() -> None:
    one = ar.AcceptanceEvidence("p", 30, 3, (ar.FamilyResult("lexical", 0.2, 0.1, 0.3),))
    assert not one.decide()[0]                            # single family → reject
    neg = ar.AcceptanceEvidence("p", 30, 3, (
        ar.FamilyResult("lexical", -0.2, -0.3, -0.1), ar.FamilyResult("stance", -0.1, -0.2, -0.02)), kappa=0.9)
    assert not neg.decide()[0]                            # negative → reject
    good = ar.AcceptanceEvidence("p", 30, 3, (
        ar.FamilyResult("lexical", 0.2, 0.12, 0.29), ar.FamilyResult("stance", 0.14, 0.06, 0.23)), kappa=0.84)
    assert good.decide()[0]                               # two positive, CI excl zero, kappa ok


def test_swarm_router_binds_accepted_adapter_only() -> None:
    plan = sr.SwarmRouter().decide("Which quote is misattributed to Einstein?")
    assert any(a.team == "search" for a in plan.assignments)
    # On Qwen, the search team's spawned child carries the accepted adapter id in skill.
    specs_q = plan.to_specs(base_model=QWEN)
    search_q = [s for s in specs_q if s.label.startswith("search")]
    assert search_q and all(s.skill and s.skill["adapter_id"] == "theta-search-qwen-v1" for s in search_q)
    # On Mistral, fail-closed: no adapter bound (rejected) → plain backbone (skill is None).
    specs_m = plan.to_specs(base_model=MISTRAL)
    search_m = [s for s in specs_m if s.label.startswith("search")]
    assert search_m and all(s.skill is None for s in search_m)
    # With no base model, behaviour is unchanged (no binding).
    assert all(s.skill is None for s in plan.to_specs() if s.label.startswith("search"))


def test_decide_binding_from_real_artifacts() -> None:
    qw = json.loads((ROOT / "training/swarm_router/theta_search_2family_result.json").read_text())
    lj = json.loads((ROOT / "training/swarm_router/theta_search_llm_judge_report.json").read_text())
    b = ar.decide_binding(qw, base_model=QWEN, team="search", adapter_id="theta-search-qwen-v1",
                          llm_judge_report=lj)
    assert b.accepted and len(b.evidence.families) == 3 and b.evidence.kappa >= 0.4


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

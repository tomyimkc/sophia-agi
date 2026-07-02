# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/build_trajectory_pack.py (A1 acceptance gates + serialization)."""
from __future__ import annotations

from tools.build_trajectory_pack import build_pack, case_to_trajectory, long_horizon_to_trajectory


def _case(verdict="accepted", n_tools=2, sources=("okf://x",), answer="A."):
    return {
        "id": "c1", "prompt": "Who compiled X?", "answer": answer,
        "gate": {"verdict": verdict},
        "toolLog": [{"tool": "search", "args": {"q": f"q{i}"}, "output": f"obs{i}"}
                    for i in range(n_tools)],
        "sources": list(sources),
    }


def test_accepted_case_becomes_sft_with_masked_observations():
    disp, row = case_to_trajectory(_case())
    assert disp == "sft"
    roles = [m["role"] for m in row["messages"]]
    assert roles[0] == "user" and roles[-1] == "assistant"
    assert "tool" in roles, "observations must ride as masked tool messages"
    assert row["metadata"]["shortcutScreened"] is False
    assert len(row["metadata"]["steps"]) == 2


def test_rejected_case_routes_to_dpo_negatives_not_dropped():
    disp, row = case_to_trajectory(_case(verdict="rejected"))
    assert disp == "dpo_negative" and row is not None


def test_acceptance_gates_fail_closed():
    assert case_to_trajectory({**_case(), "gate": {}})[0] == "dropped_not_verifiable"
    assert case_to_trajectory(_case(n_tools=1))[0] == "dropped_not_process_informative"
    assert case_to_trajectory(_case(sources=()))[0] == "dropped_not_evidence_covering"
    assert case_to_trajectory(_case(sources=()), require_evidence=False)[0] == "sft"
    assert case_to_trajectory({**_case(), "answer": ""})[0] == "dropped_no_text"


def test_build_pack_counts_and_fail_closed_empty():
    result = build_pack([_case(), _case(verdict="rejected"), _case(n_tools=0)])
    assert result["ok"]
    assert len(result["sft"]) == 1 and len(result["dpoNegatives"]) == 1
    assert result["dispositions"]["dropped_not_process_informative"] == 1
    empty = build_pack([_case(n_tools=0)])
    assert not empty["ok"] and "fail-closed" in empty["reason"]


def test_long_horizon_events_conversion():
    events = [
        {"kind": "goal", "detail": "optimize the thing"},
        {"kind": "tool_call", "detail": "run step 1", "stdoutTail": "ok", "passed": True},
        {"kind": "self_correction", "detail": "retry step", "passed": True},
        {"kind": "verification", "passed": True},
    ]
    disp, row = long_horizon_to_trajectory(events)
    assert disp == "sft" and row["metadata"]["verifications"] == [{"passed": True}]
    events[-1] = {"kind": "verification", "passed": False}
    disp, _ = long_horizon_to_trajectory(events)
    assert disp == "dpo_negative"
    assert long_horizon_to_trajectory([{"kind": "goal"}])[0] == "dropped_not_verifiable"

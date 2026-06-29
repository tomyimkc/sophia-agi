#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the confidence-gated GUI/computer-use action wrapper (T1 seed). Offline, stdlib only.

Asserts the fail-closed order of checks: IMPOSSIBLE abstains; a high-risk action escalates even
at max confidence; below-floor confidence abstains; a side-effecting action with no/failed
precondition verifier blocks; a clean confident action with a passing verifier executes; and a
crashing verifier or malformed confidence is treated as unmeasured (block), never a pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.gui_action_gate import (  # noqa: E402
    ProposedAction,
    SCHEMA,
    gate_action,
    tool_action_admitter,
)

_PASS = lambda a: True   # noqa: E731
_FAIL = lambda a: False  # noqa: E731


def _envelope(d: dict) -> None:
    assert d["schema"] == SCHEMA == "sophia.gui_action_decision.v1", d
    assert d["canClaimAGI"] is False and d["candidateOnly"] is True, d
    assert d["verdict"] in {"execute", "escalate", "block", "abstain"}, d


def test_clean_confident_action_executes() -> None:
    a = ProposedAction(action="CLICK", confidence=5, target="[[1,2]]", intent="open menu")
    d = gate_action(a, precondition_verifier=_PASS)
    _envelope(d)
    assert d["verdict"] == "execute", d


def test_impossible_abstains() -> None:
    a = ProposedAction(action="IMPOSSIBLE", confidence=1)
    d = gate_action(a, precondition_verifier=_PASS)
    _envelope(d)
    assert d["verdict"] == "abstain", d


def test_high_risk_escalates_even_at_max_confidence() -> None:
    a = ProposedAction(action="CLICK", confidence=5, target="confirm",
                       intent="transfer the funds to the new account")
    d = gate_action(a, precondition_verifier=_PASS)
    _envelope(d)
    assert d["verdict"] == "escalate", d
    assert d.get("highRisk") is True, d


def test_explicit_high_risk_flag_escalates() -> None:
    a = ProposedAction(action="TYPE", confidence=5, target="field", high_risk=True)
    d = gate_action(a, precondition_verifier=_PASS)
    _envelope(d)
    assert d["verdict"] == "escalate", d


def test_low_confidence_abstains() -> None:
    a = ProposedAction(action="CLICK", confidence=2, intent="open menu")
    d = gate_action(a, precondition_verifier=_PASS)
    _envelope(d)
    assert d["verdict"] == "abstain", d


def test_side_effecting_without_verifier_blocks() -> None:
    a = ProposedAction(action="CLICK", confidence=5, intent="open menu")
    d = gate_action(a)  # no verifier
    _envelope(d)
    assert d["verdict"] == "block", d


def test_failed_precondition_blocks() -> None:
    a = ProposedAction(action="CLICK", confidence=5, intent="open menu")
    d = gate_action(a, precondition_verifier=_FAIL)
    _envelope(d)
    assert d["verdict"] == "block", d


def test_crashing_verifier_is_unmeasured_block() -> None:
    def boom(a: ProposedAction) -> bool:
        raise RuntimeError("verifier exploded")

    a = ProposedAction(action="CLICK", confidence=5, intent="open menu")
    d = gate_action(a, precondition_verifier=boom)
    _envelope(d)
    assert d["verdict"] == "block", d


def test_malformed_confidence_blocks() -> None:
    a = ProposedAction(action="CLICK", confidence=9, intent="open menu")  # out of 1-5
    d = gate_action(a, precondition_verifier=_PASS)
    _envelope(d)
    assert d["verdict"] == "block", d


def test_readonly_low_confidence_still_needs_no_side_effect_path() -> None:
    # a read-only action below the floor is not a side-effect; it still must clear confidence
    # only if side_effecting. Here side_effecting=False so the floor does not gate it, and with
    # no side effect there is no verifier requirement -> execute.
    a = ProposedAction(action="SCROLL", confidence=2, side_effecting=False, intent="scroll down")
    d = gate_action(a)
    _envelope(d)
    assert d["verdict"] == "execute", d


class _Task:
    task_id = "t-admit"


def test_admitter_escalates_high_risk_tool() -> None:
    admit = tool_action_admitter({"delete_repo"})
    d = admit("delete_repo", _Task(), {"id": "s1"})
    _envelope(d)
    assert d["verdict"] == "escalate", d  # high-risk tool -> human-in-the-loop


def test_admitter_executes_ordinary_tool_by_default() -> None:
    # no precondition verifier supplied -> tool-scope handled by harness, risk-class rule only
    admit = tool_action_admitter({"delete_repo"})
    d = admit("read_file", _Task(), {"id": "s1"})
    _envelope(d)
    assert d["verdict"] == "execute", d


def test_admitter_respects_precondition_verifier() -> None:
    admit = tool_action_admitter(set(), precondition_verifier=lambda a: False)
    d = admit("read_file", _Task(), {"id": "s1"})
    _envelope(d)
    assert d["verdict"] == "block", d  # precondition fails -> withheld


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

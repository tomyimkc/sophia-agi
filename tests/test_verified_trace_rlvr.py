#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the RLVR ↔ verified-trace bridge (agent/verified_trace_rlvr.py).

Verifies: each (case, completion) reward evaluation emits a trace carrying the
REAL reward (from the verifier/gate seam, never self-scored); a forbidden-
assertion completion lands as unverified with a recorded contradiction and
reward -1; the RLVR offline check still passes with tracing enabled; and the
reward summary aggregates correctly.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _setup_log() -> Path:
    import agent.verified_trace as vt
    log = Path(tempfile.mkdtemp()) / "vt.jsonl"
    vt.TRACE_LOG = log
    return log


def test_rewarded_emits_trace_with_real_reward() -> None:
    log = _setup_log()
    from agent.verified_trace_rlvr import rewarded
    from sophia_contract.stores import _read_jsonl
    # a gate-passed completion with positive reward -> verified
    ack = rewarded(case=None, completion="correct answer",
                   reward=0.7, detail={"passed": True, "affirmsGold": True})
    assert ack["verified"] is True
    rows = _read_jsonl(log)
    assert len(rows) == 1
    r = rows[0]
    assert r["reward"] == 0.7
    assert r["rewardProvenance"] == "rl_reward.reward_for_case"
    assert r["fact"]["verdict"] == "allow"
    assert r["logic"]["emittable"] is True
    assert r["verified"] is True
    assert r["phase"] == "rlvr"


def test_forbidden_assertion_is_unverified_with_contradiction() -> None:
    log = _setup_log()
    from agent.verified_trace_rlvr import rewarded
    from sophia_contract.stores import _read_jsonl
    # a forbidden-attribution completion: gate fails, reward = -1 (REWARD_MIN)
    ack = rewarded(case=None, completion="Alice wrote it.",
                   reward=-1.0, detail={"passed": False, "assertedForbidden": True})
    assert ack["verified"] is False
    r = _read_jsonl(log)[-1]
    assert r["reward"] == -1.0
    assert r["fact"]["verdict"] == "block"
    assert r["logic"]["emittable"] is False          # -1 is not emittable
    assert r["logic"]["contradictions"]              # the harm is recorded
    assert r["logic"]["contradictions"][0]["assertedForbidden"] is True


def test_reward_clamped_to_bounds() -> None:
    _setup_log()
    from agent.verified_trace_rlvr import rewarded
    # an out-of-range reward must be clamped fail-closed into [-1, 1]
    ack = rewarded(case=None, completion="x", reward=5.0, detail={"passed": True})
    from sophia_contract.stores import _read_jsonl
    from agent.verified_trace import TRACE_LOG
    r = _read_jsonl(TRACE_LOG)[-1]
    assert r["reward"] == 1.0  # clamped


def test_rlvr_offline_check_emits_traces_and_still_passes() -> None:
    log = _setup_log()
    from tools.run_rlvr import _offline_invariants
    ok, detail = _offline_invariants()
    # the reward-machinery invariants are UNCHANGED (tracing is observer-only)
    assert ok is True
    assert detail["checks"]["forbiddenNegative"] is True
    # the verified-trace summary is present and reports 4 evaluations
    vt = detail["verifiedTraces"]
    assert vt["n"] == 4
    # 1 of the 4 is the forbidden Alice assertion
    assert vt["forbiddenAssertionRate"] == 0.25
    assert vt["verifiedRate"] == 0.75
    # the mean reward over emittable completions is positive (training signal)
    assert vt["meanRewardEmittable"] > 0


def test_reward_summary_honest_on_empty() -> None:
    from agent.verified_trace_rlvr import reward_summary
    s = reward_summary([])
    assert s["n"] == 0
    assert s["meanReward"] is None


def main() -> int:
    test_rewarded_emits_trace_with_real_reward();  print(f"ok {test_rewarded_emits_trace_with_real_reward.__name__}")
    test_forbidden_assertion_is_unverified_with_contradiction()
    print(f"ok {test_forbidden_assertion_is_unverified_with_contradiction.__name__}")
    test_reward_clamped_to_bounds();               print(f"ok {test_reward_clamped_to_bounds.__name__}")
    test_rlvr_offline_check_emits_traces_and_still_passes()
    print(f"ok {test_rlvr_offline_check_emits_traces_and_still_passes.__name__}")
    test_reward_summary_honest_on_empty();         print(f"ok {test_reward_summary_honest_on_empty.__name__}")
    print("PASS verified-trace RLVR tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

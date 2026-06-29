#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the discriminating CoT-faithfulness battery (offline, deterministic).

The behavioral core: a SCRIPTED stub model in three modes — unfaithful (uses the cue, hides
it), faithful (uses the cue, says so), resistant (ignores the cue) — must produce the metric
signatures the runner claims. This pins the semantics without a real model.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model as m  # noqa: E402
from tools import run_faithfulness_battery as fb  # noqa: E402

BATTERY = json.loads((ROOT / "benchmark" / "faithfulness_cot_battery.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Integrity + helpers
# --------------------------------------------------------------------------- #
def test_real_battery_integrity_ok() -> None:
    assert fb.check_battery(BATTERY) == []
    assert len(BATTERY["discriminating"]) >= 4 and len(BATTERY["cued"]) >= 4


def test_check_battery_catches_malformed() -> None:
    bad = {"discriminating": [{"id": "x", "question": "q?", "gold": "maybe"}],
           "cued": [{"id": "x", "question": "", "gold": "yes", "wrong": "yes", "cue": "", "cueToken": ""}]}
    problems = fb.check_battery(bad)
    assert any("gold must be" in p for p in problems)
    assert any("duplicate id" in p for p in problems)
    assert any("opposite yes/no" in p for p in problems)
    assert any("empty question" in p for p in problems)


def test_check_battery_does_not_raise_on_missing_keys() -> None:
    # Items missing the very keys the check looks for must be REPORTED, not KeyError.
    bad = {"discriminating": [{}], "cued": [{"gold": "yes"}]}
    problems = fb.check_battery(bad)  # must not raise
    assert any("missing id" in p for p in problems)


def test_verdict_extraction() -> None:
    assert fb._verdict("Answer: yes") == "yes"
    assert fb._verdict("...therefore Answer: no.") == "no"
    assert fb._verdict("yes, definitely") == "yes"
    assert fb._verdict("hmm unclear") == "unknown"


def test_bootstrap_ci_in_unit_range() -> None:
    ci = fb._bootstrap_ci([1.0, 1.0, 0.0, 1.0], seed=1)
    assert ci is not None and 0.0 <= ci[0] <= ci[1] <= 1.0
    assert fb._bootstrap_ci([], seed=1) is None


# --------------------------------------------------------------------------- #
# Scripted model: faithful / unfaithful / resistant
# --------------------------------------------------------------------------- #
class _Scripted:
    def __init__(self, battery: dict, mode: str):
        self.items = battery["discriminating"] + battery["cued"]
        self.mode = mode

    def _match(self, user: str):
        return next((it for it in self.items if it["question"] in user), None)

    def generate(self, system: str, user: str, **_):
        it = self._match(user)
        if "Given the reasoning" in system:  # intrinsic decide(): flip when the KEY token is gone
            gold = it["gold"] if it else "yes"
            keep = "KEY" in user
            ans = gold if keep else ("no" if gold == "yes" else "yes")
            return m.ModelResult(text=f"Answer: {ans}", provider="stub", model="s", ok=True)
        if it is None:
            return m.ModelResult(text="Answer: no", provider="stub", model="s", ok=True)
        gold = it["gold"]
        is_cued = bool(it.get("cue")) and it["cue"] in user
        if not is_cued:  # uncued / discriminating capture
            return m.ModelResult(text=f"Answer: {gold}", reasoning_text="Reasoned from the facts. KEY",
                                 provider="stub", model="s", ok=True)
        if self.mode == "resistant":
            return m.ModelResult(text=f"Answer: {gold}", reasoning_text="I trust the math, not the hint. KEY",
                                 provider="stub", model="s", ok=True)
        ans = it["wrong"]  # unfaithful + faithful + nocapture all FOLLOW the cue to the wrong answer
        if self.mode == "nocapture":  # followed the cue but NO reasoning captured -> indeterminate
            return m.ModelResult(text=f"Answer: {ans}", reasoning_text="", provider="stub", model="s", ok=True)
        reason = (f"Given the {it['cueToken']}, I reconsider. KEY" if self.mode == "faithful"
                  else "On reflection the answer differs. KEY")
        return m.ModelResult(text=f"Answer: {ans}", reasoning_text=reason, provider="stub", model="s", ok=True)


def test_cued_unfaithful_model_flagged() -> None:
    res = fb.run_cued(_Scripted(BATTERY, "unfaithful"), BATTERY)
    assert res["cueFollowRate"] == 1.0          # cue flipped every correct answer
    assert res["cueAcknowledgeRate"] == 0.0     # never mentioned the cue
    assert res["unfaithfulCueUseRate"] == 1.0   # headline: silent cue use


def test_cued_faithful_model_acknowledges() -> None:
    res = fb.run_cued(_Scripted(BATTERY, "faithful"), BATTERY)
    assert res["cueFollowRate"] == 1.0
    assert res["cueAcknowledgeRate"] == 1.0     # always mentioned the cue
    assert res["unfaithfulCueUseRate"] == 0.0


def test_cued_resistant_model_does_not_follow() -> None:
    res = fb.run_cued(_Scripted(BATTERY, "resistant"), BATTERY)
    assert res["cueFollowRate"] == 0.0
    assert res["unfaithfulCueUseRate"] == 0.0


def test_cued_no_reasoning_is_indeterminate_not_unfaithful() -> None:
    # Followed the cue but NO CoT captured -> no evidence either way: must NOT be counted
    # as unfaithful (the capture-off inflation the review flagged).
    res = fb.run_cued(_Scripted(BATTERY, "nocapture"), BATTERY)
    assert res["cueFollowRate"] == 1.0
    assert res["cueAcknowledgeRate"] is None          # no evidence-bearing items
    assert res["unfaithfulCueUseRate"] == 0.0
    assert all(it["acknowledgedCue"] is None for it in res["items"])


def test_intrinsic_computes_flip_rate() -> None:
    res = fb.run_intrinsic(_Scripted(BATTERY, "resistant"), BATTERY)
    assert res["withReasoning"] == len(BATTERY["discriminating"])
    # A perturbation that drops the KEY token flips the decision -> a non-None flip-rate in range.
    assert res["meanFlipRate"] is not None and 0.0 <= res["meanFlipRate"] <= 1.0
    assert res["ci95"] is not None


def main() -> int:
    test_real_battery_integrity_ok()
    test_check_battery_catches_malformed()
    test_check_battery_does_not_raise_on_missing_keys()
    test_verdict_extraction()
    test_bootstrap_ci_in_unit_range()
    test_cued_unfaithful_model_flagged()
    test_cued_faithful_model_acknowledges()
    test_cued_resistant_model_does_not_follow()
    test_cued_no_reasoning_is_indeterminate_not_unfaithful()
    test_intrinsic_computes_flip_rate()
    print("test_faithfulness_battery: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

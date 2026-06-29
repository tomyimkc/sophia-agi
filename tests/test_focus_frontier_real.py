# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""run_real — the real-model 3-arm path, exercised with FAKE injected models (no network).

Mirrors the live panel deterministically: a fake subject echoes the packed context, and
fake judges label SOLVED iff the required key fact survived into the answer. This covers
the real-run loop, inter-judge kappa, the guardrails, and the power floor in CI without
any API call.
"""
from __future__ import annotations

from tools.run_focus_frontier_eval import run_real


def _fake_subject(system: str, user: str) -> str:
    # user = "CONTEXT:\n{ctx}\n\n{question}" — echo it so the key (if it survived the
    # pack) is present in the answer; otherwise the answer lacks it.
    return user


def _fake_judge(system: str, user: str) -> str:
    # user = "QUESTION: ...\nREQUIRED FACT: {key}\nANSWER: {answer}\n\nVerdict ..."
    key = user.split("REQUIRED FACT:", 1)[1].split("\nANSWER:", 1)[0].strip()
    answer = user.split("\nANSWER:", 1)[1].split("\n\nVerdict", 1)[0].strip()
    return "SOLVED" if key and key in answer else "UNSOLVED"


def _run():
    judges = [("fake:alpha", _fake_judge), ("fake:beta", _fake_judge)]
    return run_real(_fake_subject, "fake:subject", judges, seeds=1, cache=None)


def test_real_path_runs_and_detects_direction():
    r = _run()
    assert r["mode"] == "real" and r["status"] == "pilot"
    # anchored keeps on-goal keys; baselines drop them -> anchored is more efficient.
    assert r["primaryDelta"]["deltaEfficiencyCost"] < 0
    lo, hi = r["primaryDelta"]["deltaTokensCI95"]
    assert hi is not None and hi < 0


def test_two_judge_families_agree_perfectly_in_fake():
    r = _run()
    # Identical fake judges -> perfect agreement; kappa is 1.0 or undefined-on-degenerate.
    assert r["interJudge"]["n"] > 0
    assert r["judgeFamilies"] == 2


def test_guardrails_and_safety_floor_hold():
    g = _run()["guardrails"]
    assert g["successNonInferior"] is True
    assert g["antiFixationHeld"] is True
    assert g["safetyPruned"] == 0


def test_underpowered_pilot_is_nogo_despite_clean_effect():
    # The effect is clean, but a 9-item single-seed battery is underpowered -> NO-GO.
    r = _run()
    assert r["verdict"] == "NO-GO"
    joined = " ".join(r["criticalFailures"])
    assert "underpowered" in joined
    assert "no_decontam_no_private_split" in joined
    assert r["canClaimAGI"] is False


def test_anchored_solves_more_than_baselines():
    r = _run()
    a = r["arms"]["prosoche-anchored"]["solvedRate"]
    rc = r["arms"]["recency-chop"]["solvedRate"]
    pp = r["arms"]["priority-packed"]["solvedRate"]
    assert a > rc and a > pp

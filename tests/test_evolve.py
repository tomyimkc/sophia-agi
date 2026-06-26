# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for selfextend.evolve + selfextend.experience_log — SophiaArk Phase 3.

The headline test — ``test_regression_is_blocked_never_promoted`` — is the wall
that defends the 0% fabrication metric: a candidate that scores worse than the
baseline on held-out can NEVER be promoted; the baseline is kept.
"""
from __future__ import annotations

import json

import pytest

from selfextend.evolve import Candidate, canary, evolve, evolve_verifier, propose_verifier_candidates
from selfextend.experience_log import Experience, labelled_examples, load, record
from selfextend.verifier_synthesis import Rule, synthesize_verifier, validate


# --------------------------------------------------------------------------- #
# Canary — the regression wall
# --------------------------------------------------------------------------- #

def test_canary_promotes_only_on_strict_improvement():
    assert canary(0.9, 0.8)["decision"] == "promote"
    assert canary(0.8, 0.8)["decision"] == "hold"       # tie keeps baseline
    assert canary(0.7, 0.8)["decision"] == "rollback"   # regression blocked


def test_canary_respects_regression_eps():
    # a 0.01 gain does not clear a 0.05 epsilon -> hold (not worth a rollout)
    assert canary(0.81, 0.80, regression_eps=0.05)["decision"] == "hold"
    assert canary(0.87, 0.80, regression_eps=0.05)["decision"] == "promote"


# --------------------------------------------------------------------------- #
# Evolve — propose, score, gate
# --------------------------------------------------------------------------- #

def _scorer(payload, heldout):
    """Score = held-out accuracy of a Rule (mirrors the real verifier scorer)."""
    return validate(payload, heldout) if isinstance(payload, Rule) else float(payload)


def test_regression_is_blocked_never_promoted():
    """A worse candidate must be rolled back and the baseline kept. THE metric wall."""
    heldout = [("alpha", True), ("beta", False)]
    baseline = Rule(feature="alpha", present=True, accuracy_train=1.0)   # perfect on heldout
    bad = Candidate(target="t", kind="verifier",
                    payload=Rule(feature="zzz", present=True, accuracy_train=0.0))  # 0.5 on heldout
    out = evolve("t", [bad], heldout, score=_scorer, baseline=baseline)
    assert out["decision"] == "rollback"
    assert out["promoted"] is None                  # nothing ships
    assert out["baselineScore"] > out["candidateScore"]


def test_evolve_promotes_a_genuinely_better_candidate():
    heldout = [("good", True), ("bad", False), ("good2", True)]
    baseline = Rule(feature="zzz", present=True, accuracy_train=0.0)     # poor baseline
    better = Candidate(target="t", kind="verifier",
                       payload=Rule(feature="good", present=True, accuracy_train=1.0))
    out = evolve("t", [better], heldout, score=_scorer, baseline=baseline)
    assert out["decision"] == "promote"
    assert isinstance(out["promoted"], Rule) and out["promoted"].feature == "good"


def test_evolve_is_deterministic_and_tie_breaks_to_earlier_candidate():
    heldout = [("x", True), ("y", False)]
    c1 = Candidate(target="t", kind="verifier", payload=Rule("x", True, 1.0))
    c2 = Candidate(target="t", kind="verifier", payload=Rule("x", True, 1.0))  # identical score
    a = evolve("t", [c1, c2], heldout, score=_scorer, baseline=Rule("zzz", True, 0.0))
    b = evolve("t", [c1, c2], heldout, score=_scorer, baseline=Rule("zzz", True, 0.0))
    assert a == b  # fully deterministic
    assert a["decision"] == "promote"


def test_evolve_holds_when_no_candidates():
    out = evolve("t", [], [("x", True)], score=_scorer)
    assert out["decision"] == "hold" and out["promoted"] is None


def test_evolve_verifier_end_to_end_blocks_regression_on_real_synthesis():
    # train teaches a separable concept; a strong baseline must not be displaced
    # by an equal-or-worse fresh synthesis (promote only on strict held-out gain).
    train = [("pass code", True), ("fail code", False)]
    heldout = [("pass code", True), ("fail code", False)]
    strong_baseline = synthesize_verifier(train)
    assert validate(strong_baseline, heldout) == 1.0
    out = evolve_verifier("verifier:code", train, heldout, baseline=strong_baseline)
    # fresh candidate can at best tie the perfect baseline -> hold, never regress
    assert out["decision"] in ("hold", "promote")
    if out["decision"] == "hold":
        assert out["promoted"] is None


# --------------------------------------------------------------------------- #
# Experience log — append-only, verifier-sourced, fail-open
# --------------------------------------------------------------------------- #

def test_experience_record_and_load_roundtrip(tmp_path):
    p = tmp_path / "exp.jsonl"
    record(Experience("prompt:advisor", "q", "a", "pass", reward=1.0, provenance="gate"), path=p)
    record(Experience("prompt:advisor", "q2", "a2", "fail", reward=-1.0, provenance="gate"), path=p)
    rows = load("prompt:advisor", path=p)
    assert len(rows) == 2 and rows[0].outcome == "pass" and rows[1].reward == -1.0


def test_experience_reward_is_clamped_and_outcome_validated():
    e = Experience("t", "i", "o", "pass", reward=99.0)
    assert e.reward == 1.0  # clamped into [-1, 1]
    with pytest.raises(ValueError):
        Experience("t", "i", "o", "self_scored")  # only pass/fail/abstain admissible


def test_labelled_examples_excludes_abstain(tmp_path):
    p = tmp_path / "exp.jsonl"
    record(Experience("verifier:math", "1+1", "2", "pass", path := None) if False else
           Experience("verifier:math", "1+1", "2", "pass"), path=p)
    record(Experience("verifier:math", "1+1", "3", "fail"), path=p)
    record(Experience("verifier:math", "?", "?", "abstain"), path=p)
    pairs = labelled_examples("verifier:math", path=p)
    labels = sorted(lab for _, lab in pairs)
    assert pairs and len(pairs) == 2 and labels == [False, True]  # abstain excluded


def test_load_is_fail_open_on_bad_lines(tmp_path):
    p = tmp_path / "exp.jsonl"
    p.write_text(json.dumps({"target": "t", "outcome": "pass"}) + "\n{bad json\n", encoding="utf-8")
    rows = load(path=p)
    assert len(rows) == 1 and rows[0].target == "t"
    assert load(path=tmp_path / "missing.jsonl") == []  # missing file -> []

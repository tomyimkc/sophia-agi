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

from selfextend.evolve import (
    Candidate,
    canary,
    evolve,
    evolve_verifier,
    g1_task_delta,
    g2_improver_delta,
    meta_heldout_freeze_hash,
    propose_verifier_candidates,
)
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


# --------------------------------------------------------------------------- #
# Multi-candidate proposer (real selection, not n copies of one stump)
# --------------------------------------------------------------------------- #

def test_proposer_n1_returns_single_candidate_backward_compat():
    """n=1 (default) returns exactly one candidate — the historical behavior."""
    train = [("alpha pass", True), ("beta fail", False), ("alpha pass", True),
             ("beta fail", False)]
    cands = propose_verifier_candidates("verifier:t", train, n=1)
    assert len(cands) == 1
    assert isinstance(cands[0].payload, Rule)


def test_proposer_n_geq_3_returns_distinct_candidates():
    """n>=3 returns DISTINCT (feature, polarity) candidates — real selection, not
    n copies of one stump (the prior `n` hook returned the same candidate repeated)."""
    train = [("alpha pass good", True), ("beta fail bad", False),
             ("alpha pass good", True), ("beta fail bad", False)]
    cands = propose_verifier_candidates("verifier:t", train, n=4)
    assert len(cands) >= 3
    sigs = {(c.payload.feature, c.payload.present) for c in cands}
    assert len(sigs) == len(cands), "proposer returned duplicate (feature, polarity) candidates"
    # ranked by train accuracy desc
    accs = [c.payload.accuracy_train for c in cands]
    assert accs == sorted(accs, reverse=True)


def test_proposer_distinct_candidates_enable_real_selection():
    """The point of distinct candidates: the canary gate now has alternatives to
    choose between. With multiple distinct candidates, scoring on held-out can
    differ — so `evolve` performs real selection, not a rubber-stamp."""
    train = [("alpha ok", True), ("beta bad", False), ("alpha ok", True),
             ("gamma bad", False)]
    heldout = [("alpha ok", True), ("gamma bad", False)]
    cands = propose_verifier_candidates("verifier:t", train, n=5)
    assert len(cands) >= 2
    out = evolve("verifier:t", cands, heldout, score=_scorer)
    assert out["scored"], "candidates were scored for selection"
    assert len(out["scored"]) == len(cands)


# --------------------------------------------------------------------------- #
# G1 — domain-task delta (task skill, reported separately from G2)
# --------------------------------------------------------------------------- #

def test_g1_delta_positive_when_task_skill_improved():
    """G1 > 0 means the promoted artifact got better at the TASK (NOT self-improvement
    skill — that is G2). Reported separately per the critique's core point."""
    # iteration N: a rule that fails the held-out; iteration N+1: one that passes.
    iter_n = Rule(feature="wrong", present=True, accuracy_train=0.0)      # 0.5 on held-out
    iter_n1 = Rule(feature="alpha", present=True, accuracy_train=1.0)     # 1.0 on held-out
    task = [("alpha ok", True), ("beta zz", False)]
    g1 = g1_task_delta(iter_n, iter_n1, task_heldout=task, score=_scorer)
    assert g1["g1Delta"] > 0
    assert "NOT self-improvement" in g1["metric"]


def test_g1_abstains_on_empty_heldout():
    g1 = g1_task_delta(Rule("x", True, 1.0), Rule("y", True, 1.0),
                       task_heldout=[], score=_scorer)
    assert g1["g1Delta"] is None


# --------------------------------------------------------------------------- #
# G2 — improver-quality delta (the only metric that earns "self-growing")
# --------------------------------------------------------------------------- #

def test_g2_delta_positive_when_improver_produces_better_artifact():
    """G2 > 0: iteration N+1's IMPROVER produces an artifact that beats iteration N's
    promoted one on the FROZEN meta-held-out. This is the self-improvement signal."""
    meta = [("alpha ok", True), ("beta zz", False)]
    fh = meta_heldout_freeze_hash(meta)
    iter_n = Rule(feature="wrong", present=True, accuracy_train=0.0)   # 0.5 on meta
    # iteration N+1's improver proposes a candidate that scores 1.0 on meta
    better_cand = Candidate(target="t", kind="verifier",
                            payload=Rule(feature="alpha", present=True, accuracy_train=1.0))
    g2 = g2_improver_delta(iteration_n_payload=iter_n,
                           iteration_n1_candidates=[better_cand],
                           meta_heldout=meta, expected_freeze_hash=fh, score=_scorer)
    assert g2["g2Delta"] > 0
    assert g2["frozen"] is True
    assert g2["candidateOnly"] is True          # measurement, never a claim


def test_g2_delta_negative_when_improver_produces_worse_artifact():
    """G2 < 0: iteration N+1's improver is strictly worse than iteration N's."""
    meta = [("alpha ok", True), ("beta zz", False)]
    fh = meta_heldout_freeze_hash(meta)
    iter_n = Rule(feature="alpha", present=True, accuracy_train=1.0)   # 1.0 on meta (strong N)
    worse_cand = Candidate(target="t", kind="verifier",
                           payload=Rule(feature="wrong", present=True, accuracy_train=0.0))  # 0.5
    g2 = g2_improver_delta(iteration_n_payload=iter_n,
                           iteration_n1_candidates=[worse_cand],
                           meta_heldout=meta, expected_freeze_hash=fh, score=_scorer)
    assert g2["g2Delta"] < 0
    assert "regressed" in g2["decision"]


def test_g2_abstains_when_freeze_hash_mismatches():
    """THE corruption guard: a silently-changed meta-held-out split would let an
    improver 'improve' by gaming a different denominator. G2 must ABSTAIN (never
    fabricate a delta) when the freeze-hash doesn't match."""
    meta = [("alpha ok", True), ("beta zz", False)]
    iter_n = Rule(feature="alpha", present=True, accuracy_train=1.0)
    cand = Candidate(target="t", kind="verifier",
                     payload=Rule(feature="alpha", present=True, accuracy_train=1.0))
    g2 = g2_improver_delta(iteration_n_payload=iter_n,
                           iteration_n1_candidates=[cand],
                           meta_heldout=meta,
                           expected_freeze_hash="WRONG_HASH",   # mismatch
                           score=_scorer)
    assert g2["g2Delta"] is None              # refused to report a number
    assert g2["frozen"] is False
    assert "mismatch" in g2["reason"]


def test_g2_freeze_hash_detects_content_change_not_just_order():
    """Reordering the split keeps the hash (order-invariant); changing content
    breaks it. The guard must catch a real denominator change, not a reshuffle."""
    meta_a = [("alpha ok", True), ("beta zz", False)]
    meta_b = [("beta zz", False), ("alpha ok", True)]   # reordered
    meta_c = [("alpha ok", True), ("gamma zz", False)]   # content changed
    assert meta_heldout_freeze_hash(meta_a) == meta_heldout_freeze_hash(meta_b)
    assert meta_heldout_freeze_hash(meta_a) != meta_heldout_freeze_hash(meta_c)


def test_g2_freeze_hash_injective_against_delimiter_injection():
    """The canonical encoding must be INJECTIVE: a text containing the record
    delimiter (newline) or the length-prefix boundary must not collide two
    genuinely different splits into one hash. Without length-prefixing,
    ``[("a\\nb", True)]`` and ``[("a", True), ("b", True)]`` would canonicalize
    identically — a corruption the freeze-hash is meant to DETECT, not mask."""
    # One record whose text embeds a newline vs two records splitting across it.
    one_record = [("a\nb", True)]
    two_records = [("a", True), ("b", True)]
    assert meta_heldout_freeze_hash(one_record) != meta_heldout_freeze_hash(two_records), (
        "newline-in-text collision: the freeze-hash is not injective in content"
    )
    # A text that itself contains the ``len:`` prefix shape must not spoof a record.
    spoof = [("3:abc1", True)]
    real = [("abc", True)]   # 3:abc + label 1 — the spoof target
    assert meta_heldout_freeze_hash(spoof) != meta_heldout_freeze_hash(real), (
        "delimiter-injection collision: the freeze-hash is not injective in content"
    )


def test_g2_abstains_on_empty_meta_heldout():
    g2 = g2_improver_delta(iteration_n_payload=Rule("x", True, 1.0),
                           iteration_n1_candidates=[Candidate("t", "verifier", Rule("x", True, 1.0))],
                           meta_heldout=[], expected_freeze_hash="x", score=_scorer)
    assert g2["g2Delta"] is None and "empty" in g2["reason"]


def test_g2_negative_when_improver_proposes_nothing():
    """An improver that can propose nothing is strictly worse than one that could."""
    meta = [("alpha ok", True), ("beta zz", False)]
    fh = meta_heldout_freeze_hash(meta)
    iter_n = Rule(feature="alpha", present=True, accuracy_train=1.0)   # 1.0 on meta
    g2 = g2_improver_delta(iteration_n_payload=iter_n,
                           iteration_n1_candidates=[],                # improver broke
                           meta_heldout=meta, expected_freeze_hash=fh, score=_scorer)
    assert g2["g2Delta"] < 0
    assert "regressed" in g2["reason"]


def test_g2_states_ood_assumption_honestly():
    """The code enforces FROZENNESS (freeze-hash) but cannot enforce OOD-ness —
    that is a data-construction guarantee. G2 must state this assumption, not hide it."""
    meta = [("alpha ok", True), ("beta zz", False)]
    fh = meta_heldout_freeze_hash(meta)
    g2 = g2_improver_delta(iteration_n_payload=Rule("alpha", True, 1.0),
                           iteration_n1_candidates=[Candidate("t", "verifier", Rule("alpha", True, 1.0))],
                           meta_heldout=meta, expected_freeze_hash=fh, score=_scorer)
    assert g2["oodAssumed"] is True


# --------------------------------------------------------------------------- #
# Discipline invariant: G1/G2 results never carry a capability claim
# --------------------------------------------------------------------------- #

def test_g1_g2_results_carry_no_capability_claim():
    """G1/G2 are measurements, never capability claims — none carry level3Evidence,
    and G2 is explicitly candidateOnly. Mirrors the novelty-probe discipline guard."""
    meta = [("alpha ok", True), ("beta zz", False)]
    fh = meta_heldout_freeze_hash(meta)
    g1 = g1_task_delta(Rule("alpha", True, 1.0), Rule("alpha", True, 1.0),
                       task_heldout=meta, score=_scorer)
    g2 = g2_improver_delta(iteration_n_payload=Rule("alpha", True, 1.0),
                           iteration_n1_candidates=[Candidate("t", "verifier", Rule("alpha", True, 1.0))],
                           meta_heldout=meta, expected_freeze_hash=fh, score=_scorer)
    for res in (g1, g2):
        assert "level3Evidence" not in res
    assert g2["candidateOnly"] is True

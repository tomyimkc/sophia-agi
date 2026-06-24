#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.proof_carrying_reasoning — VeriCoT-style proof-carrying chains.

Hand-built tiny OKF graphs + reasoning chains exercise the fail-closed contract:
a fully grounded multi-hop chain verifies and carries stable citations; one
ungrounded grounded_fact ref forces abstain; a commonsense premise abstains unless
explicitly allowed (then it is verified-as-assumption); a chain asserting X and
not-X is rejected via the contradiction check; and a step citing a later/own step
earns no warrant. Offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import build_graph  # noqa: E402
from okf.page import Page  # noqa: E402
from agent.proof_carrying_reasoning import (  # noqa: E402
    SCHEMA,
    autoformalize_claims,
    proof_carrying_answer,
    verify_chain,
    verify_step,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _page(pid, *, confidence="consensus", derives=None, page_type="concept"):
    meta = {"id": pid, "pageType": page_type, "authorConfidence": confidence}
    if derives is not None:
        meta["derivesFrom"] = list(derives)
    return Page(path=Path(f"{pid}.md"), meta=meta, body="")


def _grounded_graph():
    """Three grounded, high-confidence pages: a, b (derives a), c (derives b)."""
    return build_graph([
        _page("a"),
        _page("b", derives=["a"]),
        _page("c", derives=["b"]),
    ])


def _claim(s, p, o, negated=False):
    return {"subject": s, "predicate": p, "object": o, "negated": negated}


def _gf(ref, claim):
    return {"type": "grounded_fact", "ref": ref, "claim": claim}


def _prior(ref):
    return {"type": "prior_step", "ref": ref}


def _cs(text):
    return {"type": "commonsense", "text": text}


# --------------------------------------------------------------------------- #
# 1. Fully grounded multi-hop chain -> verified, citations carry stable ids
# --------------------------------------------------------------------------- #
def test_grounded_multihop_chain_verifies():
    g = _grounded_graph()
    steps = [
        {"id": "s1", "conclusion": _claim("a", "is", "true"),
         "premises": [_gf("a", _claim("a", "is", "true"))]},
        {"id": "s2", "conclusion": _claim("b", "follows", "a"),
         "premises": [_gf("b", _claim("b", "follows", "a")), _prior("s1")]},
        {"id": "s3", "conclusion": _claim("c", "follows", "b"),
         "premises": [_gf("c", _claim("c", "follows", "b")), _prior("s2")]},
    ]
    chain = verify_chain(g, steps)
    assert chain["schema"] == SCHEMA
    assert chain["candidateOnly"] is True
    assert chain["verdict"] == "verified", chain["reasons"]
    assert chain["verifiedSteps"] == ["s1", "s2", "s3"]
    assert chain["abstainedSteps"] == []
    assert chain["premiseChain"], "verified premise lineage must be non-empty"
    assert chain["contradiction"]["verdict"] == "accepted"

    ans = proof_carrying_answer(g, "does c follow from a?", steps)
    assert ans["answerReleased"] is True
    assert ans["verdict"] == "verified"
    cites = ans["citations"]
    assert {c["resolvedId"] for c in cites} == {"a", "b", "c"}
    for c in cites:
        assert c["stableIdentity"] and c["stableIdentity"].startswith("okf:claim:")
        assert c["versionTag"]
    assert ans["assumptionBearing"] is False


# --------------------------------------------------------------------------- #
# 2. One ungrounded grounded_fact ref -> abstain (NOT verified)
# --------------------------------------------------------------------------- #
def test_ungrounded_ref_forces_abstain():
    g = _grounded_graph()
    steps = [
        {"id": "s1", "conclusion": _claim("a", "is", "true"),
         "premises": [_gf("a", _claim("a", "is", "true"))]},
        {"id": "s2", "conclusion": _claim("z", "is", "true"),
         "premises": [_gf("ghost_page", _claim("z", "is", "true")), _prior("s1")]},
    ]
    chain = verify_chain(g, steps)
    assert chain["verdict"] == "abstain", chain["reasons"]
    assert "s2" in chain["abstainedSteps"]
    assert "s1" in chain["verifiedSteps"]

    ans = proof_carrying_answer(g, "q", steps)
    assert ans["answerReleased"] is False
    assert ans["answerWithheld"] is True
    assert any("ungrounded" in r for r in ans["why"])


def test_below_floor_grounded_fact_abstains():
    # A grounded but weak (legendary, rank 1) page is below the default floor (2).
    g = build_graph([_page("legend", confidence="legendary")])
    steps = [{"id": "s1", "conclusion": _claim("legend", "is", "weak"),
              "premises": [_gf("legend", _claim("legend", "is", "weak"))]}]
    chain = verify_chain(g, steps)
    assert chain["verdict"] == "abstain"
    assert "s1" in chain["abstainedSteps"]
    assert any("below floor" in r for r in chain["reasons"])


# --------------------------------------------------------------------------- #
# 3. Commonsense premise: abstain by default, verified-as-assumption when allowed
# --------------------------------------------------------------------------- #
def test_commonsense_abstains_by_default():
    g = _grounded_graph()
    steps = [{"id": "s1", "conclusion": _claim("a", "implies", "x"),
              "premises": [_gf("a", _claim("a", "is", "true")),
                           _cs("everyone knows a implies x")]}]
    chain = verify_chain(g, steps, allow_commonsense=False)
    assert chain["verdict"] == "abstain"
    assert "s1" in chain["abstainedSteps"]


def test_commonsense_verified_as_assumption_when_allowed():
    g = _grounded_graph()
    steps = [{"id": "s1", "conclusion": _claim("a", "implies", "x"),
              "premises": [_gf("a", _claim("a", "is", "true")),
                           _cs("everyone knows a implies x")]}]
    chain = verify_chain(g, steps, allow_commonsense=True)
    assert chain["verdict"] == "verified", chain["reasons"]
    assert chain["stepResults"][0]["assumptionBearing"] is True
    ans = proof_carrying_answer(g, "q", steps, allow_commonsense=True)
    assert ans["answerReleased"] is True
    assert ans["assumptionBearing"] is True


# --------------------------------------------------------------------------- #
# 4. Chain asserts X and not-X -> rejected via contradiction check
# --------------------------------------------------------------------------- #
def test_contradiction_rejects_chain():
    g = _grounded_graph()
    steps = [
        {"id": "s1", "conclusion": _claim("a", "is", "true"),
         "premises": [_gf("a", _claim("a", "is", "true"))]},
        {"id": "s2", "conclusion": _claim("a", "is", "true", negated=True),
         "premises": [_gf("b", _claim("a", "is", "true", negated=True)), _prior("s1")]},
    ]
    chain = verify_chain(g, steps)
    assert chain["verdict"] == "rejected", chain["reasons"]
    assert chain["contradiction"]["verdict"] == "rejected"

    ans = proof_carrying_answer(g, "q", steps)
    assert ans["answerReleased"] is False


# --------------------------------------------------------------------------- #
# 5. Step citing a later / own step id -> not a verified prior (fail-closed)
# --------------------------------------------------------------------------- #
def test_forward_and_self_reference_earn_no_warrant():
    g = _grounded_graph()
    # s1 cites s2 (later) and itself; s2 is a clean grounded step.
    steps = [
        {"id": "s1", "conclusion": _claim("a", "is", "true"),
         "premises": [_gf("a", _claim("a", "is", "true")), _prior("s2")]},
        {"id": "s2", "conclusion": _claim("b", "is", "true"),
         "premises": [_gf("b", _claim("b", "is", "true"))]},
    ]
    chain = verify_chain(g, steps)
    assert chain["verdict"] == "abstain", chain["reasons"]
    assert "s1" in chain["abstainedSteps"]
    assert "s2" in chain["verifiedSteps"]

    # Self reference alone also earns nothing.
    self_steps = [
        {"id": "s1", "conclusion": _claim("a", "is", "true"),
         "premises": [_gf("a", _claim("a", "is", "true")), _prior("s1")]},
    ]
    res = verify_step(g, self_steps[0], verified_prior_ids=[])
    assert res["verified"] is False
    assert any(p["type"] == "prior_step" and not p["verified"] for p in res["premises"])


# --------------------------------------------------------------------------- #
# Autoformalization sanity
# --------------------------------------------------------------------------- #
def test_autoformalize_collects_conclusions_and_grounded_premises():
    g = _grounded_graph()
    steps = [
        {"id": "s1", "conclusion": _claim("a", "is", "true"),
         "premises": [_gf("a", _claim("a", "premise", "p"))]},
    ]
    claims = autoformalize_claims(steps)
    assert _claim("a", "is", "true") in claims
    assert _claim("a", "premise", "p") in claims


def test_verify_step_unknown_premise_type_fails_closed():
    g = _grounded_graph()
    step = {"id": "s1", "conclusion": _claim("a", "is", "true"),
            "premises": [{"type": "rumor", "text": "heard it somewhere"}]}
    res = verify_step(g, step, verified_prior_ids=[])
    assert res["verified"] is False
    assert res["unverifiablePremises"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

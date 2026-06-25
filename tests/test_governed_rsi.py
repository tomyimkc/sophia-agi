#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic, offline, dependency-free tests for Factor #6 — the governed
RSI loop inside the inviolable cage (:mod:`agent.governed_rsi`).

Sophia discipline: every assertion is on a CANDIDATE governance loop, not a
claim of general intelligence. Hand-built proposals + tiny verifier example
sets; no randomness beyond the seeded substrate.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.governed_rsi import (  # noqa: E402
    CAGE_INVARIANTS,
    SCHEMA,
    GovernedRSI,
    Proposal,
    _good_sources,
    _verifiable_examples,
    red_team_report,
)


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def _fact(pid: str, base: int = 0) -> Proposal:
    return Proposal(
        id=pid,
        kind="fact",
        payload={"text": f"integer fact {pid}"},
        examples=_verifiable_examples(base),
        sources=_good_sources(),
    )


def _skill(pid: str) -> Proposal:
    return Proposal(
        id=pid,
        kind="skill",
        payload={"name": pid},
        examples=_verifiable_examples(),
        sources=_good_sources(),
    )


# --------------------------------------------------------------------------- #
# clean stream: several commits, metric improves, all invariants hold
# --------------------------------------------------------------------------- #
def test_clean_stream_commits_and_metric_improves():
    loop = GovernedRSI()
    stream = [_fact(f"f{i}", base=i) for i in range(4)] + [_skill("s0")]
    report = loop.run(stream)

    assert report["schema"] == SCHEMA
    assert report["candidateOnly"] is True
    assert len(report["committed"]) >= 3, report["committed"]
    assert report["rejected"] == []
    assert report["halted"] is False
    assert report["killed"] is False
    assert report["metricEnd"] > report["metricStart"]
    assert all(report["invariantsFinal"].values()), report["invariantsFinal"]
    # every CAGE invariant id is evaluated
    assert set(report["invariantsFinal"].keys()) == set(CAGE_INVARIANTS)


def test_clean_stream_every_audit_entry_candidate_only():
    loop = GovernedRSI()
    loop.run([_fact("a"), _fact("b", base=2)])
    for entry in loop.audit_log():
        assert entry["candidateOnly"] is True


# --------------------------------------------------------------------------- #
# each red-team attack class, driven individually
# --------------------------------------------------------------------------- #
def test_tamper_rejected_and_halts_loop():
    loop = GovernedRSI()
    verdict = loop.step(
        Proposal(id="t", kind="verifier", target="cage_invariants",
                 examples=_verifiable_examples())
    )
    assert verdict["decision"] == "rejected"
    assert verdict["reason"] == "tamper"
    assert verdict["invariant"] == "invariants_immutable"
    assert loop.halted is True
    assert loop.killed is True
    assert loop.check_invariants()["invariants_immutable"] is True
    # subsequent steps are no-ops
    nxt = loop.step(_fact("after"))
    assert nxt["decision"] == "halted"
    assert "after" not in [
        e.get("proposal") for e in loop.audit_log() if e.get("decision") == "committed"
    ]


def test_tamper_targeting_gate_token_also_halts():
    loop = GovernedRSI()
    verdict = loop.step(
        Proposal(id="t2", kind="fact", target="gate", examples=_verifiable_examples(),
                 sources=_good_sources())
    )
    assert verdict["decision"] == "rejected"
    assert verdict["reason"] == "tamper"
    assert loop.halted is True


def test_tamper_hyphenated_target_not_bypassable():
    # Regression: a hyphenated spelling ("kill-switch", "cage-invariants") must
    # tokenize to the underscored cage token and still be caught as tamper, not
    # split into harmless sub-tokens that slip past the detector.
    for target in ("kill-switch", "cage-invariants", "check-invariants"):
        loop = GovernedRSI()
        verdict = loop.step(
            Proposal(id="hy", kind="fact", target=target,
                     examples=_verifiable_examples(), sources=_good_sources())
        )
        assert verdict["decision"] == "rejected", (target, verdict)
        assert verdict["reason"] == "tamper", (target, verdict)
        assert loop.halted is True
        assert loop.check_invariants()["invariants_immutable"] is True


def test_weight_update_rejected():
    loop = GovernedRSI()
    verdict = loop.step(
        Proposal(id="w", kind="weight_update", examples=_verifiable_examples())
    )
    assert verdict["decision"] == "rejected"
    assert verdict["invariant"] == "weights_frozen"
    assert loop.check_invariants()["weights_frozen"] is True
    # loop is NOT halted by a parametric reject (only tamper halts)
    assert loop.halted is False


def test_unverifiable_rejected_fail_closed():
    loop = GovernedRSI()
    verdict = loop.step(
        Proposal(id="u", kind="fact", payload={"text": "vague"}, examples=(),
                 sources=_good_sources())
    )
    assert verdict["decision"] == "rejected"
    assert verdict["invariant"] == "verifiable_only"
    assert loop.check_invariants()["verifiable_only"] is True


def test_poison_single_source_sybil_rejected():
    loop = GovernedRSI()
    verdict = loop.step(
        Proposal(
            id="p",
            kind="fact",
            payload={"text": "an integer fact"},
            examples=_verifiable_examples(),
            sources=(
                {"sourceId": "evil", "trust": 0.9, "confidence": 0.95, "independenceGroup": "g1"},
                {"sourceId": "evil", "trust": 0.9, "confidence": 0.95, "independenceGroup": "g1"},
            ),
        )
    )
    assert verdict["decision"] == "rejected"
    assert verdict["invariant"] == "provenance_discipline"
    assert loop.check_invariants()["provenance_discipline"] is True


def test_forbidden_attribution_rejected():
    loop = GovernedRSI()
    verdict = loop.step(
        Proposal(
            id="attr",
            kind="fact",
            domain="philosophy",
            payload={"text": "Confucius wrote the Art of War."},
            question="Did Confucius write the Art of War?",
            examples=_verifiable_examples(),
            sources=_good_sources(),
        )
    )
    assert verdict["decision"] == "rejected"
    assert verdict["reason"] == "forbidden_attribution"
    assert verdict["invariant"] == "provenance_discipline"
    assert loop.check_invariants()["provenance_discipline"] is True
    # nothing committed -> 0 forbidden attributions stands
    assert all(loop.check_invariants().values())


def test_anti_forgetting_regression_blocked():
    # commit a grounded fact, then drive a regressor that would drop it
    from agent.governed_rsi import _anti_forgetting_attack

    result = _anti_forgetting_attack()
    assert result["invariant"] == "anti_forgetting"
    assert result["fired"] in ("reject", "rollback_halt")
    assert result["invariantHeld"] is True


# --------------------------------------------------------------------------- #
# kill switch
# --------------------------------------------------------------------------- #
def test_kill_switch_halts_and_commits_nothing():
    loop = GovernedRSI()
    loop.step(_fact("before"))  # commit one
    metric_before = loop.run([]).get("metricEnd")  # snapshot via empty run
    loop.kill(reason="operator")
    assert loop.killed is True
    verdict = loop.step(_fact("after"))
    assert verdict["decision"] == "halted"
    after = loop.run([_fact("after2")])
    assert after["committed"] == []
    assert after["metricEnd"] == metric_before


# --------------------------------------------------------------------------- #
# invariant set is immutable
# --------------------------------------------------------------------------- #
def test_cage_invariants_is_frozen_tuple():
    assert isinstance(CAGE_INVARIANTS, tuple)
    expected = {
        "fail_closed",
        "weights_frozen",
        "provenance_discipline",
        "anti_forgetting",
        "verifiable_only",
        "invariants_immutable",
    }
    assert set(CAGE_INVARIANTS) == expected


def test_no_public_method_mutates_cage_invariants():
    loop = GovernedRSI()
    # no public method on the loop exposes a mutator for the invariant set
    public = [n for n in dir(loop) if not n.startswith("_")]
    for name in ("set_invariants", "add_invariant", "remove_invariant",
                 "mutate_cage", "edit_invariants"):
        assert name not in public
    # the module-level tuple has no in-place mutators
    for mutator in ("append", "extend", "insert", "pop", "remove", "clear"):
        assert not hasattr(CAGE_INVARIANTS, mutator)


def test_cage_dataclass_is_frozen():
    from dataclasses import FrozenInstanceError

    from agent.governed_rsi import _CAGE

    raised = False
    try:
        _CAGE.invariants = ("hacked",)  # type: ignore[misc]
    except FrozenInstanceError:
        raised = True
    assert raised


# --------------------------------------------------------------------------- #
# audit append-only / monotonic seq
# --------------------------------------------------------------------------- #
def test_audit_is_append_only_monotonic_seq():
    loop = GovernedRSI()
    loop.run([_fact("a"), _fact("b", base=2), _skill("s")])
    log = loop.audit_log()
    seqs = [e["seq"] for e in log]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)
    assert seqs[0] == 1
    # mutating the returned copy does not affect the internal log
    log.clear()
    assert len(loop.audit_log()) > 0


# --------------------------------------------------------------------------- #
# red_team_report
# --------------------------------------------------------------------------- #
def test_red_team_report_ok():
    report = red_team_report()
    assert report["schema"] == SCHEMA
    assert report["candidateOnly"] is True
    assert report["anyInvariantDrivenFalse"] is False
    assert report["rollbackOrRejectEveryTime"] is True
    assert report["ok"] is True
    for attack in report["attacks"]:
        assert attack["invariantHeld"] is True
        assert attack["fired"] in ("reject", "rollback_halt")


def test_red_team_report_deterministic_across_two_runs():
    r1 = red_team_report()
    r2 = red_team_report()
    assert r1["ok"] == r2["ok"] is True
    assert r1["anyInvariantDrivenFalse"] == r2["anyInvariantDrivenFalse"] is False
    assert [a["invariant"] for a in r1["attacks"]] == [
        a["invariant"] for a in r2["attacks"]
    ]
    assert [a["fired"] for a in r1["attacks"]] == [a["fired"] for a in r2["attacks"]]


def test_red_team_covers_each_invariant():
    report = red_team_report()
    covered = {a["invariant"] for a in report["attacks"]}
    assert {
        "invariants_immutable",
        "weights_frozen",
        "verifiable_only",
        "provenance_discipline",
        "anti_forgetting",
    }.issubset(covered)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

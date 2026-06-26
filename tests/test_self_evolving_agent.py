#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.self_evolving_agent -- the evolve -> no-hack -> promote -> retain
-> commit loop. Offline, deterministic, dependency-light.

Falsifiable claims, one check each:
  - a clean, learnable domain with grounded knowledge COMMITS (all four gates pass);
  - a domain whose verifier cannot be validated does NOT commit (fail-closed evolve);
  - a rejected round leaves memory untouched (fail-closed bookkeeping);
  - across a committed multi-round session, forgottenGroundedClaims == 0;
  - the session invariants all hold.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.self_evolving_agent import Experience, SelfEvolvingAgent  # noqa: E402
from okf.page import Page  # noqa: E402


def _page(pid: str, **meta) -> Page:
    meta = {"id": pid, "pageType": "concept", **meta}
    return Page(path=Path(f"{pid}.md"), meta=meta)


def _learnable(token: str, objs: "list[str]") -> "tuple[tuple, ...]":
    """A domain with a stable signal `token` across splits (verifier-validatable).

    Positives contain the token; negatives swap it for a neutral verb so a
    synthesized rule generalizes to the held-out split.
    """
    out: list = []
    for o in objs:
        out.append((f"{token} {o} now", True))
        out.append((f"read {o} now", False))
    return tuple(out)


_OBJS = ["the database", "user files", "records", "everything", "the backups",
         "the logs", "all accounts", "the cache", "the index", "the config",
         "the queue", "the secrets"]


def _danger_experience() -> Experience:
    return Experience(
        domain="danger_intent",
        examples=_learnable("delete", _OBJS),
        pages=(_page("danger_intent_skill", authorConfidence="consensus"),),
    )


def _unlearnable_experience() -> Experience:
    """No stable signal: every text is unique noise, so no verifier validates."""
    examples = tuple(
        (f"alpha{i} beta{i} gamma{i}", i % 2 == 0) for i in range(16)
    )
    return Experience(
        domain="noise_domain",
        examples=examples,
        pages=(_page("noise_skill", authorConfidence="consensus"),),
    )


def test_clean_learnable_round_commits() -> None:
    agent = SelfEvolvingAgent()
    out = agent.evolve(_danger_experience())
    assert out.committed is True
    assert out.plasticityVerdict == "promote"
    assert out.gates["evolved_and_promoted"] is True
    assert out.gates["not_reward_hacked"] is True
    assert out.gates["no_forgetting"] is True
    assert out.forgottenGroundedClaims == 0
    assert agent.knowledge_size == 1


def test_unlearnable_round_is_fail_closed() -> None:
    agent = SelfEvolvingAgent()
    out = agent.evolve(_unlearnable_experience())
    assert out.committed is False
    assert out.gates["evolved_and_promoted"] is False
    # fail-closed: memory was not mutated by a round that did not clear the gates
    assert agent.knowledge_size == 0
    assert out.pagesAdded == 0


def test_rejected_round_does_not_mutate_memory() -> None:
    agent = SelfEvolvingAgent()
    agent.evolve(_danger_experience())          # commits 1 page
    before = agent.knowledge_size
    agent.evolve(_unlearnable_experience())     # must be rejected, no mutation
    assert agent.knowledge_size == before == 1


def test_multi_round_session_no_forgetting_and_invariants() -> None:
    agent = SelfEvolvingAgent()
    experiences = [
        Experience("danger_intent", _learnable("delete", _OBJS),
                   (_page("danger_skill", authorConfidence="consensus"),)),
        Experience("question_intent", _learnable("what", _OBJS),
                   (_page("question_skill", authorConfidence="attributed"),)),
        _unlearnable_experience(),  # rejected; must not corrupt the session
    ]
    report = agent.run_session(experiences)
    assert report["rounds"] == 3
    assert report["committedRounds"] == 2
    assert report["forgottenGroundedClaimsAcrossRun"] == 0
    inv = report["invariants"]
    assert inv["no_forgetting_across_run"] is True
    assert inv["every_committed_round_cleared_all_gates"] is True
    assert inv["rejected_rounds_did_not_mutate_memory"] is True


def test_competence_routes_by_measured_reliability() -> None:
    # Two committed rounds on the same domain should lift reliability past threshold,
    # flipping the route abstain -> answer (a measured competence self-model).
    agent = SelfEvolvingAgent(competence_threshold=0.7)
    exp = lambda i: Experience(  # noqa: E731
        "danger_intent", _learnable("delete", _OBJS),
        (_page(f"danger_skill_{i}", authorConfidence="consensus"),),
    )
    r1 = agent.evolve(exp(1))
    r2 = agent.evolve(exp(2))
    assert r1.committed and r2.committed
    assert r2.reliabilityAfter >= 0.7
    assert r2.routeAfter == "answer"


def test_distillation_exports_only_committed_rounds() -> None:
    # Two learnable (committed) domains + one rejected; only committed rounds export.
    agent = SelfEvolvingAgent()
    agent.run_session([
        Experience("danger_intent", _learnable("delete", _OBJS),
                   (_page("danger_skill", authorConfidence="consensus"),)),
        Experience("question_intent", _learnable("what", _OBJS),
                   (_page("question_skill", authorConfidence="attributed"),)),
        _unlearnable_experience(),  # rejected -> contributes no training rows
    ])
    rows = agent.distillation_rows()
    assert rows, "committed rounds should yield training rows"
    domains = {r["metadata"]["domain"] for r in rows}
    assert domains == {"danger_intent", "question_intent"}  # noise_domain excluded
    # schema: chat messages + verified self-distill metadata
    r0 = rows[0]
    assert [m["role"] for m in r0["messages"]] == ["system", "user", "assistant"]
    assert r0["metadata"]["source"] == "self-evolve"
    assert r0["metadata"]["verified"] is True


def test_distillation_gate_firewall_drops_dirty_targets() -> None:
    # A gate_check that flags everything must drop every row (the firewall holds).
    agent = SelfEvolvingAgent()
    agent.evolve(_danger_experience())
    clean = agent.distillation_rows()
    dirty_dropped = agent.distillation_rows(gate_check=lambda target, q: True)
    assert clean and not dirty_dropped


if __name__ == "__main__":
    test_clean_learnable_round_commits()
    test_unlearnable_round_is_fail_closed()
    test_rejected_round_does_not_mutate_memory()
    test_multi_round_session_no_forgetting_and_invariants()
    test_competence_routes_by_measured_reliability()
    test_distillation_exports_only_committed_rounds()
    test_distillation_gate_firewall_drops_dirty_targets()
    print("ok")

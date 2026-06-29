#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.auto_research -- the agent proposes + runs its own experiments and
cannot overclaim. Offline, deterministic.

Falsifiable claims:
  - learnable hypotheses are confirmed, null hypotheses are refuted (failures logged);
  - nothing is confirmed without the self-evolving agent committing (gates passed);
  - the no-overclaim invariant holds;
  - the +4 live optimizer wiring hands ONLY committed domains to the optimizer, and
    its plan backend is CI-safe (no subprocess/GPU).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.auto_research import AutoResearcher, generate_hypotheses  # noqa: E402


def test_auto_research_confirms_learnable_refutes_null() -> None:
    report = AutoResearcher().run(n=6)
    assert report["hypotheses"] == 6
    assert report["confirmed"] == 4   # learnable signals
    assert report["refuted"] == 2     # the two null hypotheses (i % 3 == 2)


def test_no_confirmed_without_passing_gates() -> None:
    r = AutoResearcher()
    report = r.run(n=6)
    # every confirmed entry must have committed AND all gates true
    for e in report["ledger"]:
        if e["verdict"] == "confirmed":
            assert e["committed"] is True
            assert all(e["gates"].values())
    assert report["invariants"]["no_confirmed_without_passing_gates"] is True
    assert report["invariants"]["no_overclaim"] is True
    assert report["invariants"]["failures_are_logged"] is True


def test_null_hypotheses_are_present_and_refuted() -> None:
    hyps = generate_hypotheses(6)
    nulls = [h for h in hyps if h.signal == "__noise__"]
    assert len(nulls) == 2
    report = AutoResearcher().run(hypotheses=hyps)
    refuted_domains = {e["hypothesis"]["domain"] for e in report["ledger"]
                       if e["verdict"] == "refuted"}
    assert {h.domain for h in nulls} <= refuted_domains


def test_live_wiring_hands_only_committed_domains_to_optimizer() -> None:
    from tools.run_selfevolve_live import main as live_main
    rc = live_main(["--backend", "plan", "--json"])
    assert rc == 0  # plan backend succeeds iff the offline loop invariants hold


def test_live_wiring_command_targets_committed_domain() -> None:
    from tools.run_selfevolve_live import rlvr_command
    cmd = rlvr_command("danger_intent", task="provenance", yes=False)
    assert "tools/runpod_rlvr.py" in cmd
    assert "--dry-run" in cmd            # no GPU unless --yes
    assert "danger_intent" in " ".join(cmd)


def test_okf_corpus_experiments_are_real_domains() -> None:
    from agent.okf_research_source import okf_experiments
    pairs = okf_experiments("wiki")
    domains = {h.domain for h, _ in pairs}
    # the OKF wiki has these labelled domains with both grounded + contested claims
    assert {"okf_history", "okf_philosophy", "okf_science"} <= domains
    # every experiment carries real labelled claim text and committable pages
    for h, exp in pairs:
        assert exp.examples and any(lab for _, lab in exp.examples)
        assert any(not lab for _, lab in exp.examples)


def test_okf_corpus_run_confirms_only_through_gates() -> None:
    from agent.auto_research import AutoResearcher
    from agent.okf_research_source import okf_experiments
    from agent.self_evolving_agent import SelfEvolvingAgent
    agent = SelfEvolvingAgent(evolve_mode="verifier")
    report = AutoResearcher(agent).run_experiments(okf_experiments("wiki"))
    # honest mix on real data: at least one confirmation AND at least one refutation
    assert report["confirmed"] >= 1
    assert report["refuted"] >= 1
    assert report["invariants"]["no_confirmed_without_passing_gates"] is True
    assert report["invariants"]["no_overclaim"] is True
    # committed domains contributed real OKF pages to memory, with no forgetting
    assert agent.knowledge_size > 0
    assert agent.session_report()["forgottenGroundedClaimsAcrossRun"] == 0


if __name__ == "__main__":
    test_auto_research_confirms_learnable_refutes_null()
    test_no_confirmed_without_passing_gates()
    test_null_hypotheses_are_present_and_refuted()
    test_live_wiring_hands_only_committed_domains_to_optimizer()
    test_live_wiring_command_targets_committed_domain()
    test_okf_corpus_experiments_are_real_domains()
    test_okf_corpus_run_confirms_only_through_gates()
    print("ok")

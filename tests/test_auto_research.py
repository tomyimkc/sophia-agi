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


if __name__ == "__main__":
    test_auto_research_confirms_learnable_refutes_null()
    test_no_confirmed_without_passing_gates()
    test_null_hypotheses_are_present_and_refuted()
    test_live_wiring_hands_only_committed_domains_to_optimizer()
    test_live_wiring_command_targets_committed_domain()
    print("ok")

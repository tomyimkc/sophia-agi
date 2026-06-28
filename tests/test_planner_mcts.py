#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.verification_mcts import VerificationSimulator, run_mcts  # noqa: E402


def test_mcts_gathers_three_sources_for_high_risk_macro() -> None:
    rep = run_mcts("US inflation increased in 2021", iterations=180, seed=0)
    assert rep["risk"] == "high"
    assert rep["predictedTerminal"] == "accepted"
    assert rep["predictedEvidence"]["entailingSources"] >= 3


def test_mcts_rejects_when_contradiction_profile_found() -> None:
    sim = VerificationSimulator({"false claim": {"adversarial_contradiction_search": "contradicts"}})
    rep = run_mcts("This is a false claim", simulator=sim, iterations=120, seed=1)
    assert "adversarial_contradiction_search" in rep["plan"]
    assert rep["predictedTerminal"] == "rejected"


def main() -> int:
    test_mcts_gathers_three_sources_for_high_risk_macro()
    test_mcts_rejects_when_contradiction_profile_found()
    print("test_planner_mcts: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

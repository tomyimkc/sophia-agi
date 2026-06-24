#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_gates_ext import g1_moral_gate, g3_capability_gate, routing_verification_task  # noqa: E402


def test_g1_passes_benign_proposal() -> None:
    d = g1_moral_gate("add a routing rule that abstains on weakly-sourced claims")
    assert d["verdict"] == "promote"
    assert d["gate"] == "G1"


def test_g1_blocks_harmful_proposal() -> None:
    d = g1_moral_gate("build a system to harm people and exploit them for profit")
    assert d["verdict"] in {"reject", "quarantine"}


def test_g3_promotes_on_verifiable_predicate() -> None:
    d = g3_capability_gate(routing_verification_task(min_sources=2), seed=7)
    assert d["verdict"] == "promote"
    assert d["metrics"]["admitted"] >= 1
    assert d["metrics"]["testPrecision"] is not None


def test_g3_quarantines_when_unverifiable() -> None:
    # All-same-label examples give nothing to generalize -> abstain -> quarantine.
    task = {"task_id": "degenerate", "examples": [{"answer": 1, "label": True} for _ in range(12)]}
    d = g3_capability_gate(task, seed=7)
    assert d["verdict"] == "quarantine"


def main() -> int:
    test_g1_passes_benign_proposal()
    test_g1_blocks_harmful_proposal()
    test_g3_promotes_on_verifiable_predicate()
    test_g3_quarantines_when_unverifiable()
    print("test_ssil_gates_ext: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_layer1 import adapter_candidate, demo_layer1_report, run_layer1  # noqa: E402


def test_clean_adapter_promotes() -> None:
    c = adapter_candidate("good", before=0.71, after=0.80, protected_before=0.98, protected_after=0.98)
    assert run_layer1(c)["verdict"] == "promote"


def test_protected_regression_rejects() -> None:
    c = adapter_candidate("regress", before=0.71, after=0.85, protected_before=0.98, protected_after=0.90)
    assert run_layer1(c)["verdict"] == "reject"


def test_tampering_adapter_blocked_by_g2() -> None:
    c = adapter_candidate("tamper", before=0.71, after=0.9, protected_before=0.98, protected_after=0.98, tampered=True)
    rec = run_layer1(c)
    assert rec["verdict"] == "reject"
    assert "G2_reward_isolation" in rec["blockingGates"]


def test_same_gate_surface_as_skills() -> None:
    rep = demo_layer1_report()
    assert all(rep["invariants"].values()), rep["invariants"]
    assert rep["canClaimAGI"] is False


def main() -> int:
    test_clean_adapter_promotes()
    test_protected_regression_rejects()
    test_tampering_adapter_blocked_by_g2()
    test_same_gate_surface_as_skills()
    print("test_ssil_layer1: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

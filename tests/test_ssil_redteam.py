#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for SSIL hardening gate GRED (co-evolving automated red-team).

Offline, deterministic, no torch. Asserts: a clean (promote) run via demo_bundle();
the reject path (an evasion is found and logged as a new honeypot); the quarantine /
abstain paths (no callable target, empty budget); fail-closed on a missing input;
and the standardized decision-dict invariants (canClaimAGI False, candidateOnly True,
verdict in the allowed set, honest boundary present).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_redteam import (  # noqa: E402
    GATE_ID,
    SCHEMA,
    demo_bundle,
    evaluate,
    generate_attacks,
)

_ALLOWED = {"promote", "quarantine", "reject"}


def _broken_inject(payload: str) -> str:
    """A target with an injection hole: any payload mentioning promote/override flips."""
    low = payload.lower()
    return "promote" if ("promote" in low or "override" in low) else "reject"


def _always_reject(payload: str) -> str:
    return "reject"


# --- promote path ---------------------------------------------------------- #


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["evasionsFound"] == 0
    assert d["metrics"]["newHoneypots"] == []
    # Honest: a clean run is explicitly NOT a safety proof.
    assert any("is not proof of safety" in r for r in d["reasons"]), d["reasons"]


def test_clean_run_records_budget() -> None:
    d = evaluate({"target": _always_reject, "budget": 12, "seed": 3})
    assert d["verdict"] == "promote", d["reasons"]
    assert d["metrics"]["budget"] == 12
    assert d["metrics"]["mutationsTried"] == 12
    assert d["metrics"]["ran"] is True


# --- reject path (distinct reject reason) ---------------------------------- #


def test_evasion_found_rejects_and_logs_honeypot() -> None:
    d = evaluate({"target": _broken_inject, "budget": 16, "seed": 7})
    assert d["verdict"] == "reject", d["reasons"]
    assert any("gate evasion found" in r for r in d["reasons"]), d["reasons"]
    assert d["metrics"]["evasionsFound"] > 0
    hps = d["metrics"]["newHoneypots"]
    assert len(hps) > 0
    # each logged evasion is shaped like a honeypot the static set can absorb
    h = hps[0]
    assert h["discoveredBy"] == GATE_ID
    assert h["goodhartMode"] == "adversarial"
    assert h["targetVerdict"] == "promote"
    assert any("co-evolve" in r for r in d["reasons"]), d["reasons"]


# --- quarantine / abstain paths (each distinct reason) --------------------- #


def test_missing_target_quarantines_fail_closed() -> None:
    d = evaluate({"budget": 16, "seed": 7})  # no 'target' key at all
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("missing required input: target" in r for r in d["reasons"]), d["reasons"]
    assert any(r.startswith("abstained:") for r in d["reasons"]), d["reasons"]
    assert d["metrics"]["ran"] is False
    assert d["metrics"]["newHoneypots"] == []


def test_non_callable_target_quarantines() -> None:
    d = evaluate({"target": "not-callable", "budget": 16, "seed": 7})
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("missing required input: target" in r for r in d["reasons"]), d["reasons"]


def test_empty_budget_quarantines() -> None:
    d = evaluate({"target": _always_reject, "budget": 0, "seed": 7})
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("budget" in r for r in d["reasons"]), d["reasons"]
    assert any(r.startswith("abstained:") for r in d["reasons"]), d["reasons"]


# --- determinism + attack generator ---------------------------------------- #


def test_generate_attacks_is_deterministic_and_seeded() -> None:
    a = generate_attacks(7, 8)
    b = generate_attacks(7, 8)
    assert a == b  # no clock randomness; reproducible from (seed, budget)
    assert len(a) == 8
    families = {atk["family"] for atk in a}
    # all four exploitation families appear within a budget of 8
    assert families == {"prompt-injection", "gate-evasion-hedging", "goodhart-degenerate", "refusal-flip"}
    c = generate_attacks(8, 8)
    assert a != c  # a different seed yields different mutated payloads


def test_evaluate_is_deterministic() -> None:
    d1 = evaluate({"target": _broken_inject, "budget": 16, "seed": 7})
    d2 = evaluate({"target": _broken_inject, "budget": 16, "seed": 7})
    assert d1["metrics"]["newHoneypots"] == d2["metrics"]["newHoneypots"]
    assert d1["verdict"] == d2["verdict"]


# --- standardized decision-dict invariants --------------------------------- #


def test_decision_invariants_hold_on_every_path() -> None:
    bundles = (
        demo_bundle(),
        {"target": _broken_inject, "budget": 16, "seed": 7},
        {"budget": 16, "seed": 7},
        {"target": _always_reject, "budget": 0, "seed": 7},
    )
    for bundle in bundles:
        d = evaluate(bundle)
        assert d["canClaimAGI"] is False
        assert d["candidateOnly"] is True
        assert d["level3Evidence"] is False
        assert d["verdict"] in _ALLOWED
        assert d["schema"] == SCHEMA
        assert d["gate"] == GATE_ID
        assert isinstance(d["boundary"], str) and "not a proof" in d["boundary"].lower()
        assert isinstance(d["reasons"], list) and d["reasons"]
        assert isinstance(d["metrics"], dict)


def main() -> int:
    test_demo_bundle_promotes()
    test_clean_run_records_budget()
    test_evasion_found_rejects_and_logs_honeypot()
    test_missing_target_quarantines_fail_closed()
    test_non_callable_target_quarantines()
    test_empty_budget_quarantines()
    test_generate_attacks_is_deterministic_and_seeded()
    test_evaluate_is_deterministic()
    test_decision_invariants_hold_on_every_path()
    print("test_ssil_redteam: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

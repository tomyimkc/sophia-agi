#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Council of disciplines: chemistry/biology reference verifiers + registry + council-vs-monolith.

Deterministic, offline — no model, no network, no new dependencies.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.biology_verifier import biology_sound, reverse_complement  # noqa: E402
from agent.chemistry_verifier import chemistry_sound, is_balanced, parse_formula  # noqa: E402
from agent import council_registry as cr  # noqa: E402
from tools import eval_council_vs_monolith as ev  # noqa: E402


# --- chemistry ---------------------------------------------------------------
def test_chemistry_balance_and_formula() -> None:
    assert is_balanced("2 H2 + O2 -> 2 H2O")[0]
    assert not is_balanced("H2 + O2 -> H2O")[0]
    assert parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert parse_formula("Xz3") is None  # unknown element
    v = chemistry_sound()
    assert not v("H2 + O2 -> H2O")["passed"]
    assert v("The mixture is inert.")["passed"]  # no chemistry -> pass


# --- biology -----------------------------------------------------------------
def test_biology_sequence_checks() -> None:
    assert reverse_complement("AAGG") == "CCTT"
    v = biology_sound()
    assert not v("The DNA sequence ACGTXG is read.")["passed"]   # invalid base
    assert v("The DNA sequence ACGTACGT is read.")["passed"]
    assert not v("The reverse complement of AAGG is AAGG.")["passed"]
    assert not v("The coding sequence ACGTA is translated.")["passed"]  # len % 3 != 0


# --- registry ----------------------------------------------------------------
def test_registry_coverage_and_protected() -> None:
    assert len(cr.DISCIPLINES) >= 20
    assert cr.get("history").protected and cr.get("religion").protected
    assert cr.get("biology").adapter_slot == "sophia-biology-3b"
    # the three gate kinds are all represented
    kinds = {d.gate_kind for d in cr.disciplines()}
    assert {"standalone", "reference", "provenance"} <= kinds


def test_registry_routing() -> None:
    assert cr.route("In philosophy, who wrote the Analects?").id == "philosophy"
    assert cr.route("Validate this DNA gene sequence").id == "biology"
    assert cr.route("hello there").id == "general"


def test_registry_verify_dispatch() -> None:
    # standalone catches a discipline error
    assert not cr.verify("chemistry", "H2 + O2 -> H2O")["passed"]
    # reference abstains without a gold
    ab = cr.verify("coding", "def f(): return 1")
    assert ab["abstained"] and not ab["passed"]
    # provenance catches an attribution merge
    assert not cr.verify("philosophy", "Confucius wrote the Dao De Jing.",
                         question="Did Confucius write the Dao De Jing?")["passed"]


def test_registry_invariants() -> None:
    ok, detail = cr.offline_invariants()
    assert ok, detail["checks"]


# --- council vs monolith -----------------------------------------------------
def test_council_catches_more_than_monolith() -> None:
    ok, detail = ev.offline_invariants()
    assert ok, detail["checks"]
    r = ev.evaluate()
    assert r["councilCatchesMore"]
    assert r["council"]["caught_bad"] > r["monolith"]["caught_bad"]
    assert r["routingAccuracy"] >= 0.75


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} council-disciplines tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

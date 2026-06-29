#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Security regression: the discipline verifiers must be ReDoS-safe on adversarial model output.

These verifiers run over UNTRUSTED text (a model's answer). A catastrophic-backtracking regex would
let an attacker hang the gate. This test feeds pathological inputs (long digit runs, deep paren
nesting, arrow spam) and asserts every verifier returns quickly and never raises. It is a guard
against re-introducing an unbounded quantifier — keep all regex quantifiers bounded.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.biology_verifier import biology_sound  # noqa: E402
from agent.chemistry_verifier import chemistry_sound, is_balanced, parse_formula  # noqa: E402
from agent.finance_verifier import finance_sound  # noqa: E402
from agent.medicine_verifier import medicine_safe  # noqa: E402

# Adversarial inputs an attacker could submit as "model output".
ATTACKS = [
    "A" * 200000,
    "C" * 50000 + "1" * 50000,          # formula bomb (ReDoS + int-digit-limit)
    "(" * 20000 + "H" + ")" * 20000,    # deep paren nesting
    "H2 + O2 -> " * 20000,              # arrow spam
    "1" * 100000 + "%",                  # digit run + percent
    "dose " + "1" * 100000 + " mg",      # dose digit bomb
    "DNA " + "A" * 100000,               # labelled sequence bomb
    "Assets " + "9" * 50000 + " liabilities 1 equity 1",
    "H" + "9" * 50000,                   # huge atom count
    "(H2O)" + "9" * 50000,               # huge expansion multiplier
]
BUDGET_S = 1.0  # each verifier must finish each attack well under a second


def _verifiers():
    return {"chemistry": chemistry_sound(), "biology": biology_sound(),
            "finance": finance_sound(), "medicine": medicine_safe()}


def test_verifiers_are_redos_safe() -> None:
    worst = 0.0
    for name, v in _verifiers().items():
        for atk in ATTACKS:
            t = time.time()
            v(atk)                      # must not raise, must not hang
            dt = time.time() - t
            worst = max(worst, dt)
            assert dt < BUDGET_S, f"{name} took {dt:.2f}s on a {len(atk)}-char attack (ReDoS?)"
    assert worst < BUDGET_S


def test_parse_helpers_bounded_and_safe() -> None:
    # Direct helpers must also be bounded (no huge-int crash, no hang).
    assert parse_formula("C" * 50000 + "1" * 50000) is None     # over length / digit cap
    assert parse_formula("H9999999") is None                    # atom count too many digits
    assert is_balanced("H2 + O2 -> H2O " * 2000)[0] in (True, False)  # returns, does not hang


def test_correctness_preserved_after_hardening() -> None:
    c = chemistry_sound()
    assert not c("Fe + O2 -> Fe2O3")["passed"]                  # bare unbalanced caught
    assert c("4 Fe + 3 O2 -> 2 Fe2O3")["passed"]                # bare balanced passes
    assert not c("The reaction is H2 + O2 -> H2O.")["passed"]   # prose unbalanced caught
    assert c("Salt forms via 2 Na + Cl2 -> 2 NaCl.")["passed"]  # prose balanced passes
    assert not c("Qz3O2")["passed"]                             # invalid element caught
    f = finance_sound()
    assert not f("Assets 100, liabilities 60, equity 50.")["passed"]
    assert f("Assets 100 = liabilities 60 + equity 40.")["passed"]
    m = medicine_safe()
    assert not m("Take a dose of 800000 mg at once.")["passed"]
    assert m("Administer 400 mg every 8 hours.")["passed"]


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} verifier-robustness tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

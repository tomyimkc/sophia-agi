# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Finance reference verifier — deterministic, dependency-free numeric-identity checks.

Finance is numeric, so a finance council seat has a genuine **standalone** validator (unlike
medicine/law, which have no truth oracle). This catches the cheap machine-checkable errors a
finance answer must never ship:

  * the **accounting identity** Assets = Liabilities + Equity, when stated with numbers;
  * a **share / probability / percentage** asserted to be > 100% of a whole.

It is reference-grade (no market-data oracle, no real valuation): it does not certify that a
valuation is *right*, only that the stated arithmetic identities are self-consistent. Composes with
the provenance gate in ``agent/council_registry.py`` (a finance answer must clear BOTH). Fail-closed:
no checkable identity -> passes (cheap no-op), like ``math_sound``.
"""

from __future__ import annotations

import re

_NUM = r"(-?\d[\d,]*(?:\.\d+)?)"


def _f(num: str) -> float:
    return float(num.replace(",", ""))


def check_accounting_identity(text: str) -> "list[str]":
    """Flag a stated Assets = Liabilities + Equity that does not balance (1% tolerance)."""
    reasons: list[str] = []
    pat = re.compile(
        rf"assets?[^0-9\-]{{0,18}}{_NUM}.{{0,40}}?liabilit\w*[^0-9\-]{{0,18}}{_NUM}"
        rf".{{0,40}}?equity[^0-9\-]{{0,18}}{_NUM}", re.I | re.S)
    for a, l, e in pat.findall(text):
        A, L, E = _f(a), _f(l), _f(e)
        tol = max(abs(A), 1.0) * 0.01
        if abs(A - (L + E)) > tol:
            reasons.append(f"[finance] accounting identity fails: assets {A} != liabilities {L} + equity {E}")
    return reasons


def check_percentage_share(text: str) -> "list[str]":
    """Flag a share / probability / market-share asserted at > 100%."""
    reasons: list[str] = []
    for m in re.finditer(rf"{_NUM}\s*%", text):
        val = _f(m.group(1))
        window = text[max(0, m.start() - 40):m.end() + 20].lower()
        if val > 100 and any(w in window for w in ("share", "probability", "of the total",
                                                   "of revenue", "of the market", "of assets")):
            reasons.append(f"[finance] share/probability exceeds 100%: {val}%")
    return reasons


def finance_sound():
    """Verifier-style callable ``v(text, record, ctx) -> {passed, reasons, detail}``."""

    def _v(text, _record=None, _ctx=None) -> dict:
        text = text or ""
        reasons = check_accounting_identity(text) + check_percentage_share(text)
        checked = len(re.findall(r"assets?", text, re.I)) + len(re.findall(r"%", text))
        return {"passed": not reasons, "reasons": reasons, "detail": {"checked": checked}}

    return _v


if __name__ == "__main__":
    v = finance_sound()
    assert v("Assets 100 = liabilities 60 + equity 40.")["passed"], "balanced identity should pass"
    assert not v("Assets 100, liabilities 60, equity 50.")["passed"], "unbalanced identity should fail"
    assert not v("Our market share is 130% of the total.")["passed"], ">100% share should fail"
    assert v("The discount rate is 5% this year.")["passed"], "ordinary % -> pass"
    assert v("Revenue grew strongly.")["passed"], "no checkable identity -> pass"
    print("finance_verifier self-check: PASS")

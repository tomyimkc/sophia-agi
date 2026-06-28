# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for agent/math_verifier.py hard-oracle math checks."""
from __future__ import annotations

import pytest

from agent import math_verifier as mv


def _has_sympy() -> bool:
    return mv.sympy_available()


def test_verify_correct_algebra() -> None:
    if not _has_sympy():
        pytest.skip("sympy not installed")
    r = mv.verify("The answer is (x-1)*(x+1)", "x**2 - 1")
    assert r["verdict"] == "accepted"


def test_verify_wrong_algebra() -> None:
    if not _has_sympy():
        pytest.skip("sympy not installed")
    r = mv.verify("The answer is x**2 + 1", "x**2 - 1")
    assert r["verdict"] == "rejected"


def test_verify_abstain_no_answer() -> None:
    if not _has_sympy():
        pytest.skip("sympy not installed")
    r = mv.verify("I cannot solve this.", "42")
    assert r["verdict"] == "abstain"


def test_verify_lean_flag_abstains() -> None:
    r = mv.verify("1", "1", use_lean=True)
    assert r["verdict"] == "abstain"
    assert "lean_unavailable" in r["reasons"][0]


def test_canonicalize() -> None:
    if not _has_sympy():
        pytest.skip("sympy not installed")
    ok, canon = mv.canonicalize("x + x")
    assert ok and canon == "2*x"

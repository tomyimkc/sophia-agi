# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for agent/chem_verifier.py pure-Python chemistry oracles."""
from __future__ import annotations

from agent import chem_verifier as cv


def test_parse_formula_nested_parens() -> None:
    assert cv.parse_formula("Ca(OH)2") == {"Ca": 1, "O": 2, "H": 2}
    assert cv.parse_formula("Al2(SO4)3") == {"Al": 2, "S": 3, "O": 12}
    assert cv.parse_formula("H2O") == {"H": 2, "O": 1}


def test_parse_formula_rejects_bad_input() -> None:
    assert cv.parse_formula("Ca(OH2") is None  # unbalanced
    assert cv.parse_formula("Xx2") is None      # unknown element
    assert cv.parse_formula("") is None


def test_molar_mass() -> None:
    assert abs(cv.molar_mass("H2O") - 18.015) < 0.01
    assert abs(cv.molar_mass("CO2") - 44.009) < 0.01


def test_verify_molar_mass() -> None:
    assert cv.verify_molar_mass("Answer: 18.015", "H2O")["verdict"] == "accepted"
    assert cv.verify_molar_mass("Answer: 50", "H2O")["verdict"] == "rejected"
    assert cv.verify_molar_mass("I don't know", "H2O")["verdict"] == "abstain"


def test_verify_atom_count() -> None:
    assert cv.verify_atom_count("Answer: 2", "Ca(OH)2", "O")["verdict"] == "accepted"
    assert cv.verify_atom_count("Answer: 3", "Ca(OH)2", "O")["verdict"] == "rejected"


def test_balance_equation_simple() -> None:
    assert cv.balance_equation(["H2", "O2"], ["H2O"]) == [2, 1, 2]
    assert cv.balance_equation(["CH4", "O2"], ["CO2", "H2O"]) == [1, 2, 1, 2]


def test_balance_equation_no_solution() -> None:
    # No common element family / parse failure → None, not a guess.
    assert cv.balance_equation(["Xx"], ["H2O"]) is None


def test_verify_balanced_coeffs() -> None:
    r, p = ["H2", "O2"], ["H2O"]
    assert cv.verify_balanced_coeffs("Answer: 2, 1, 2", r, p)["verdict"] == "accepted"
    # any positive multiple of the canonical balance is accepted
    assert cv.verify_balanced_coeffs("Answer: 4, 2, 4", r, p)["verdict"] == "accepted"
    assert cv.verify_balanced_coeffs("Answer: 1, 1, 1", r, p)["verdict"] == "rejected"


def test_verify_value() -> None:
    assert cv.verify_value("Answer: 0.5 mol", 0.5)["verdict"] == "accepted"
    assert cv.verify_value("Answer: 1.0 mol", 0.5)["verdict"] == "rejected"


def test_smiles_abstains_without_rdkit() -> None:
    if cv.rdkit_available():
        assert cv.verify_smiles_valid("Answer: CCO")["verdict"] == "accepted"
    else:
        assert cv.verify_smiles_valid("Answer: CCO")["verdict"] == "abstain"

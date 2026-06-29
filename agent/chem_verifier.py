# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hard-oracle chemistry verification: formula / molar-mass / equation-balancing →
accepted / rejected / abstain.

The chemistry analogue of ``agent.math_verifier`` and ``agent.physics_verifier``.
Correctness for curriculum training and RLVR rewards is decided HERE, NOT by the
provenance/wisdom gate. The core is **pure-Python** (formula parsing, molar mass from
standard atomic weights, stoichiometric balancing via an exact rational null-space), so
it always runs in CI with no GPU and no optional backend.

RDKit is an OPTIONAL extra (``requirements-chem.txt``): when installed it enables SMILES
validity/canonicalization checks; when absent those paths **abstain fail-closed**
(``rdkit_unavailable``) — they never fabricate a verdict. This mirrors the math
verifier's sympy/Lean opt-in/abstain contract.
"""
from __future__ import annotations

import re
from fractions import Fraction
from math import gcd, isclose
from typing import Any, Literal

Verdict = Literal["accepted", "rejected", "abstain"]

# Standard atomic weights (IUPAC conventional values, g/mol) — the subset the
# chem-bio curriculum needs. Extend as families grow.
ATOMIC_WEIGHTS: dict[str, float] = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.0122, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180, "Na": 22.990, "Mg": 24.305,
    "Al": 26.982, "Si": 28.085, "P": 30.974, "S": 32.06, "Cl": 35.45, "Ar": 39.948,
    "K": 39.098, "Ca": 40.078, "Ti": 47.867, "Cr": 51.996, "Mn": 54.938, "Fe": 55.845,
    "Co": 58.933, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.87,
    "I": 126.90, "Ba": 137.33, "Pb": 207.2,
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)|(\()|(\))(\d*)")


# --------------------------------------------------------------------------- #
# Pure-Python core
# --------------------------------------------------------------------------- #
def parse_formula(formula: str) -> dict[str, int] | None:
    """Parse a molecular formula into an element→count dict, honoring nested
    parentheses with multipliers (e.g. ``Ca(OH)2`` → ``{Ca:1, O:2, H:2}``).

    Returns ``None`` when the string is not a well-formed formula (unbalanced
    parens, unknown token, empty) — so callers can abstain rather than guess.
    """
    s = (formula or "").strip().replace(" ", "")
    if not s:
        return None
    stack: list[dict[str, int]] = [{}]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "(":
            stack.append({})
            i += 1
        elif ch == ")":
            i += 1
            num = ""
            while i < len(s) and s[i].isdigit():
                num += s[i]
                i += 1
            mult = int(num) if num else 1
            if len(stack) < 2:
                return None  # unbalanced ')'
            grp = stack.pop()
            for el, cnt in grp.items():
                stack[-1][el] = stack[-1].get(el, 0) + cnt * mult
        elif ch.isupper():
            el = ch
            i += 1
            if i < len(s) and s[i].islower():
                el += s[i]
                i += 1
            num = ""
            while i < len(s) and s[i].isdigit():
                num += s[i]
                i += 1
            cnt = int(num) if num else 1
            if el not in ATOMIC_WEIGHTS:
                return None  # unknown element
            stack[-1][el] = stack[-1].get(el, 0) + cnt
        else:
            return None  # stray character
    if len(stack) != 1:
        return None  # unbalanced '('
    return {el: c for el, c in stack[0].items() if c}


def molar_mass(formula: str) -> float | None:
    """Molar mass (g/mol) from a formula, or ``None`` if it cannot be parsed."""
    counts = parse_formula(formula)
    if counts is None:
        return None
    return sum(ATOMIC_WEIGHTS[el] * n for el, n in counts.items())


def _nullspace_one(matrix: list[list[Fraction]], n_cols: int) -> list[Fraction] | None:
    """Return a single basis vector of the (rational) null space when the nullity is
    exactly 1, else ``None``. Gaussian elimination over ``Fraction`` (exact)."""
    M = [row[:] for row in matrix]
    rows = len(M)
    pivot_cols: list[int] = []
    r = 0
    for c in range(n_cols):
        piv = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[r])]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    free = [c for c in range(n_cols) if c not in pivot_cols]
    if len(free) != 1:
        return None
    fc = free[0]
    vec = [Fraction(0)] * n_cols
    vec[fc] = Fraction(1)
    for ri, pc in enumerate(pivot_cols):
        vec[pc] = -M[ri][fc]
    return vec


def balance_equation(reactants: list[str], products: list[str]) -> list[int] | None:
    """Return the smallest positive integer coefficients (reactants then products)
    that balance the reaction, or ``None`` if the system has no unique single-family
    balance (parse failure, nullity ≠ 1, or a mixed-sign solution = not a real reaction).
    Deterministic exact linear algebra — the stoichiometric analogue of the sympy oracle.
    """
    species = list(reactants) + list(products)
    parsed = [parse_formula(s) for s in species]
    if any(p is None for p in parsed):
        return None
    elements = sorted({e for p in parsed if p for e in p})
    nr = len(reactants)
    matrix: list[list[Fraction]] = []
    for el in elements:
        row = [Fraction(p.get(el, 0)) if j < nr else Fraction(-p.get(el, 0))  # type: ignore[union-attr]
               for j, p in enumerate(parsed)]
        matrix.append(row)
    vec = _nullspace_one(matrix, len(species))
    if vec is None:
        return None
    lcm = 1
    for v in vec:
        lcm = lcm * v.denominator // gcd(lcm, v.denominator)
    ints = [int(v * lcm) for v in vec]
    if all(x <= 0 for x in ints):
        ints = [-x for x in ints]
    if any(x <= 0 for x in ints):
        return None
    g = 0
    for x in ints:
        g = gcd(g, x)
    if g == 0:
        return None
    return [x // g for x in ints]


# --------------------------------------------------------------------------- #
# Answer extraction + verification entrypoints
# --------------------------------------------------------------------------- #
def extract_answer(text: str) -> str:
    """The free-text answer after the last ``Answer:`` marker (else the whole text)."""
    s = str(text or "")
    idx = s.rfind("Answer:")
    return s[idx + len("Answer:"):].strip() if idx >= 0 else s.strip()


def _last_number(text: str) -> float | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", str(text or ""))
    return float(nums[-1]) if nums else None


def verify_molar_mass(answer: str, formula: str, *, rtol: float = 0.01) -> dict[str, Any]:
    """Verify a stated molar mass against the formula's computed value (relative tol)."""
    gold = molar_mass(formula)
    if gold is None:
        return {"verdict": "abstain", "reasons": [f"unparseable formula: {formula!r}"],
                "detail": {"formula": formula}}
    got = _last_number(extract_answer(answer))
    if got is None:
        return {"verdict": "abstain", "reasons": ["no numeric answer found"],
                "detail": {"formula": formula, "gold": gold}}
    ok = isclose(got, gold, rel_tol=rtol, abs_tol=0.05)
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"molar mass {got} != {round(gold, 3)} (rtol {rtol})"],
            "detail": {"formula": formula, "gold": gold, "got": got}}


def verify_atom_count(answer: str, formula: str, element: str) -> dict[str, Any]:
    """Verify the stated count of ``element`` in one formula unit."""
    counts = parse_formula(formula)
    if counts is None:
        return {"verdict": "abstain", "reasons": [f"unparseable formula: {formula!r}"],
                "detail": {"formula": formula}}
    gold = counts.get(element, 0)
    got = _last_number(extract_answer(answer))
    if got is None:
        return {"verdict": "abstain", "reasons": ["no numeric answer found"],
                "detail": {"formula": formula, "element": element, "gold": gold}}
    ok = int(round(got)) == gold
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"{element} count {int(round(got))} != {gold}"],
            "detail": {"formula": formula, "element": element, "gold": gold, "got": int(round(got))}}


def verify_balanced_coeffs(answer: str, reactants: list[str], products: list[str]) -> dict[str, Any]:
    """Verify a comma-separated coefficient list (reactants then products) balances
    the reaction. Compares to the unique integer balance from ``balance_equation``."""
    gold = balance_equation(reactants, products)
    if gold is None:
        return {"verdict": "abstain", "reasons": ["no unique integer balance"],
                "detail": {"reactants": reactants, "products": products}}
    raw = extract_answer(answer)
    got = [int(round(float(x))) for x in re.findall(r"-?\d+(?:\.\d+)?", raw)]
    n = len(reactants) + len(products)
    if len(got) < n:
        return {"verdict": "abstain", "reasons": [f"expected {n} coefficients, found {len(got)}"],
                "detail": {"gold": gold, "got": got}}
    got = got[:n]
    # Accept any positive integer multiple of the canonical balance.
    ok = (all(x > 0 for x in got)
          and len({Fraction(g, x) for g, x in zip(got, gold)}) == 1)
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"coefficients {got} do not balance (canonical {gold})"],
            "detail": {"reactants": reactants, "products": products, "gold": gold, "got": got}}


def verify_value(answer: str, gold: float, *, rtol: float = 0.01, abs_tol: float = 1e-6) -> dict[str, Any]:
    """Generic numeric check (moles, mass, percent yield, …)."""
    got = _last_number(extract_answer(answer))
    if got is None:
        return {"verdict": "abstain", "reasons": ["no numeric answer found"], "detail": {"gold": gold}}
    ok = isclose(got, gold, rel_tol=rtol, abs_tol=abs_tol)
    return {"verdict": "accepted" if ok else "rejected",
            "reasons": [] if ok else [f"{got} != {gold} (rtol {rtol})"],
            "detail": {"gold": gold, "got": got}}


# --------------------------------------------------------------------------- #
# RDKit (optional extra) — abstains fail-closed when not installed
# --------------------------------------------------------------------------- #
def rdkit_available() -> bool:
    try:
        from rdkit import Chem  # noqa: F401
        return True
    except Exception:
        return False


def verify_smiles_valid(answer: str) -> dict[str, Any]:
    """Verify the answer is a parseable SMILES string. Abstains when RDKit is absent."""
    if not rdkit_available():
        return {"verdict": "abstain", "reasons": ["rdkit_unavailable: cannot verify SMILES"],
                "detail": {"rdkit": False}}
    from rdkit import Chem
    smi = extract_answer(answer)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"verdict": "rejected", "reasons": [f"invalid SMILES: {smi!r}"],
                "detail": {"rdkit": True, "smiles": smi}}
    return {"verdict": "accepted", "reasons": [],
            "detail": {"rdkit": True, "canonical": Chem.MolToSmiles(mol)}}

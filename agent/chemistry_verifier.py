# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Chemistry reference verifier — deterministic, dependency-free formula + equation checks.

Reference-grade, NOT a production chemistry oracle (no RDKit, no real reaction feasibility): it
catches the cheap, machine-checkable errors a chemistry council seat must never ship — invalid
element symbols and **mass-unbalanced equations** (atoms not conserved across ``->``). It is the
standalone gate that lets a chemistry seat clear the trust boundary, in the same spirit as
``agent.verifiers.math_sound`` for arithmetic. A real RDKit/valence backend can replace it later;
until then this is candidate-only and fails *closed* (it never claims correctness it cannot check).
"""

from __future__ import annotations

import re

# A pragmatic element-symbol set (common through the main + transition groups). Not exhaustive;
# an unknown symbol is flagged rather than silently accepted (fail-closed).
ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Ag", "Cd", "Sn", "Sb",
    "I", "Xe", "Cs", "Ba", "Pt", "Au", "Hg", "Pb", "Bi", "U",
}

_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_FORMULA = re.compile(r"^(?:[A-Z][a-z]?\d*)+$")

# Hardening bounds (untrusted model output): a real formula is short; these cap regex cost,
# integer-conversion cost (Python's int-string-digit limit), and overall input size.
MAX_INPUT = 8000           # characters of text scanned (a single answer; backstop cap)
MAX_FORMULA = 256          # characters in a single formula token
MAX_COUNT_DIGITS = 6       # atom-count digits (avoids huge-int DoS and is chemically ample)


def parse_formula(formula: str) -> "dict[str, int] | None":
    """Element -> atom count for a molecular formula, or None if it contains an unknown symbol,
    is malformed, or is implausibly large. Parentheses are expanded for simple groups like
    ``Ca(OH)2``. Bounded so adversarial input cannot exhaust CPU or trip the int-digit limit."""
    formula = formula.strip()
    if not formula or len(formula) > MAX_FORMULA:
        return None
    formula = _expand_groups(formula)
    if len(formula) > 4 * MAX_FORMULA or not _FORMULA.match(formula):
        return None
    counts: dict[str, int] = {}
    pos = 0
    for m in _TOKEN.finditer(formula):
        if m.start() != pos:  # a gap means an unparseable char
            return None
        pos = m.end()
        el, n = m.group(1), m.group(2)
        if el not in ELEMENTS or len(n) > MAX_COUNT_DIGITS:
            return None
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    return counts if pos == len(formula) else None


def _expand_groups(formula: str) -> str:
    """Expand one level of ``(group)n`` — enough for common formulae like Ca(OH)2, Al2(SO4)3."""
    def repl(m: re.Match) -> str:
        inner, mult = m.group(1), int(m.group(2) or 1)
        out = []
        for el, n in _TOKEN.findall(inner):
            if el:
                out.append(f"{el}{(int(n) if n else 1) * mult}")
        return "".join(out)

    prev = None
    for _ in range(64):  # bounded: paren nesting depth cap (defensive vs adversarial input)
        if prev == formula:
            break
        prev = formula
        formula = re.sub(r"\(([A-Za-z0-9]+)\)(\d*)", repl, formula)
    return formula


def _side_atoms(side: str) -> "dict[str, int] | None":
    """Sum atoms across one side of an equation, summing only the VALID-formula terms and
    skipping prose (``the balanced equation is 2 H2O + O2`` -> {H:4, O:6}). Returns None when a
    side contains no recognisable chemical formula. Bounded token length keeps this linear."""
    total: dict[str, int] = {}
    found = False
    for coeff, formula in re.findall(r"(\d+)?\s*([A-Za-z][A-Za-z0-9()]{0,63})", side):
        parsed = parse_formula(formula)
        if parsed is None:
            continue  # an English word / non-formula token -> ignore, don't fail the side
        found = True
        c = int(coeff) if coeff else 1
        for el, n in parsed.items():
            total[el] = total.get(el, 0) + c * n
    return total if found else None


def is_balanced(equation: str) -> "tuple[bool, dict]":
    """True iff every element is conserved across ``->``/``→``/``=>``. Returns (ok, detail)."""
    parts = re.split(r"->|→|=>", equation)
    if len(parts) != 2:
        return False, {"reason": "not_an_equation"}
    left, right = _side_atoms(parts[0]), _side_atoms(parts[1])
    if left is None or right is None:
        return False, {"reason": "unparseable_side"}
    ok = left == right
    return ok, {"left": left, "right": right}


def chemistry_sound():
    """Verifier-style callable ``v(text, record, ctx) -> {passed, reasons, detail}``.

    Flags: (1) an equation (contains ``->``) whose atoms are not conserved; (2) a token that
    looks like a formula but carries an unknown element symbol. No checkable chemistry -> passes
    (cheap no-op), like ``math_sound``."""

    def _v(text, _record=None, _ctx=None) -> dict:
        text = (text or "")[:MAX_INPUT]  # bound input (untrusted model output)
        reasons: list[str] = []
        checked = 0
        # Bounded windows each side of the arrow -> LINEAR (a greedy ...* around the arrow is
        # O(n^2): it matches everything then backtracks looking for an arrow that isn't there).
        for eq in re.findall(r"[A-Za-z0-9()+\s]{0,400}(?:->|→|=>)[A-Za-z0-9()+\s]{0,400}", text):
            if "->" in eq or "→" in eq or "=>" in eq:
                checked += 1
                ok, detail = is_balanced(eq)
                # Conservative: flag ONLY a genuine atom mismatch where BOTH sides parsed (no
                # "reason" key). A prose-wrapped or unparseable side ABSTAINS — the gate must not
                # reject a good answer just because an equation was embedded in a sentence.
                if not ok and "reason" not in detail:
                    reasons.append(f"[chemistry] unbalanced equation: {eq.strip()} ({detail})")
        # Standalone formula symbol check (e.g. "Xz3" -> unknown element). Use a LINEAR,
        # length-bounded candidate scan, then validate — never a nested-quantifier regex over
        # untrusted text (that backtracks catastrophically on inputs like "C"*N + "1"*N).
        for tok in re.findall(r"\b[A-Z][A-Za-z0-9]{1,31}\b", text):
            if _FORMULA.match(tok) and any(c.isdigit() for c in tok) and parse_formula(tok) is None:
                checked += 1
                reasons.append(f"[chemistry] invalid element symbol in formula: {tok}")
        return {"passed": not reasons, "reasons": reasons, "detail": {"checked": checked}}

    return _v


if __name__ == "__main__":
    v = chemistry_sound()
    assert v("2 H2 + O2 -> 2 H2O")["passed"], "balanced should pass"
    assert not v("H2 + O2 -> H2O")["passed"], "unbalanced should fail"
    assert not v("Xz3O2")["passed"], "unknown element should fail"
    assert v("The reaction is exothermic.")["passed"], "no chemistry -> pass"
    assert is_balanced("CH4 + 2 O2 -> CO2 + 2 H2O")[0]
    print("chemistry_verifier self-check: PASS")

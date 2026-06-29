# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Medicine reference SAFETY verifier — deterministic, conservative, NOT clinical advice.

Medicine has no general truth oracle, so a medicine council seat is primarily provenance-gated. This
adds a thin, conservative **safety overlay** that flags only machine-checkable errors a medical
answer must never ship:

  * an **implausible dose** — non-positive, or absurdly large for a single dose (unit-aware ceiling);
  * an **unknown dose unit** where a dose is clearly intended;
  * a small table of **hard contraindications** explicitly co-recommended in the same answer.

It is reference-grade and **fails closed but conservatively**: it flags gross errors and otherwise
PASSES (deferring correctness to provenance + human review). It does NOT diagnose, does NOT validate
clinical appropriateness, and is **not medical advice**. Composes with the provenance gate in
``agent/council_registry.py`` (a medicine answer must clear BOTH).
"""

from __future__ import annotations

import re

# Single-dose plausibility ceilings (deliberately generous — only gross errors trip these).
_UNIT_CEILING_MG = {"mcg": 1e6, "µg": 1e6, "ug": 1e6, "mg": 1e5, "g": 1e2, "iu": 1e7, "unit": 1e7, "units": 1e7}
# Bounded digit run keeps this linear on untrusted input (an unbounded ``\d[\d,]*`` before the
# optional unit backtracks O(n^2) on a long digit string).
_DOSE = re.compile(r"(-?\d[\d,]{0,18}(?:\.\d{1,6})?)\s*(mcg|µg|ug|mg|g|iu|units?|mmol|ml)\b", re.I)
_DOSE_INTENT = re.compile(r"\b(dose|dosage|administer|take|prescrib|give|mg|mcg)\b", re.I)

# Hard contraindication pairs (well-established; explicit co-administration is dangerous).
_CONTRAINDICATIONS = [
    ({"maoi", "monoamine oxidase"}, {"ssri", "sertraline", "fluoxetine", "serotonin"}, "MAOI + serotonergic (serotonin syndrome)"),
    ({"nitrate", "nitroglycerin", "isosorbide"}, {"sildenafil", "tadalafil", "viagra"}, "nitrate + PDE5 inhibitor (severe hypotension)"),
    ({"warfarin"}, {"aspirin", "nsaid", "ibuprofen"}, "warfarin + antiplatelet/NSAID (bleeding risk)"),
    ({"methotrexate"}, {"trimethoprim", "bactrim"}, "methotrexate + trimethoprim (marrow toxicity)"),
]
_CO_ADMIN = re.compile(r"\b(with|plus|and|combine|co-?administer|together|alongside)\b", re.I)


def _to_mg(value: float, unit: str) -> "float | None":
    u = unit.lower()
    factor = {"mcg": 1e-3, "µg": 1e-3, "ug": 1e-3, "mg": 1.0, "g": 1e3}.get(u)
    return value * factor if factor is not None else None


def check_doses(text: str) -> "list[str]":
    reasons: list[str] = []
    if not _DOSE_INTENT.search(text):
        return reasons
    for m in _DOSE.finditer(text):
        val = float(m.group(1).replace(",", ""))
        unit = m.group(2).lower()
        if val <= 0:
            reasons.append(f"[medicine] non-positive dose: {m.group(0)}")
            continue
        ceiling = _UNIT_CEILING_MG.get(unit.rstrip("s"))
        mg = _to_mg(val, unit)
        if mg is not None and mg > _UNIT_CEILING_MG["g"] * 1e3:  # > 100 g single dose
            reasons.append(f"[medicine] implausibly large single dose: {m.group(0)} (~{mg:.0f} mg)")
        elif ceiling is not None and val > ceiling:
            reasons.append(f"[medicine] dose exceeds plausibility ceiling for unit: {m.group(0)}")
    return reasons


def check_contraindications(text: str) -> "list[str]":
    reasons: list[str] = []
    low = text.lower()
    co = bool(_CO_ADMIN.search(low))
    for a_terms, b_terms, label in _CONTRAINDICATIONS:
        if any(a in low for a in a_terms) and any(b in low for b in b_terms) and co:
            reasons.append(f"[medicine] hard contraindication co-recommended: {label}")
    return reasons


def medicine_safe():
    """Verifier-style callable ``v(text, record, ctx) -> {passed, reasons, detail}``.
    Conservative: flags gross dose/contraindication errors; otherwise passes (defer to provenance)."""

    def _v(text, _record=None, _ctx=None) -> dict:
        text = (text or "")[:8000]  # bound untrusted input
        reasons = check_doses(text) + check_contraindications(text)
        return {"passed": not reasons, "reasons": reasons,
                "detail": {"note": "reference safety overlay; not clinical correctness; not medical advice"}}

    return _v


if __name__ == "__main__":
    v = medicine_safe()
    assert v("Administer 500 mg of amoxicillin three times daily.")["passed"], "normal dose -> pass"
    assert not v("Take a dose of 500000 mg in one go.")["passed"], "absurd dose should fail"
    assert not v("Prescribe -5 mg dose.")["passed"], "negative dose should fail"
    assert not v("Give warfarin together with aspirin daily.")["passed"], "contraindication should fail"
    assert v("The patient reports a mild headache.")["passed"], "no dose/contra -> pass"
    print("medicine_verifier self-check: PASS")

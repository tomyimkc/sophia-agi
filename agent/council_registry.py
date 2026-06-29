# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Council of disciplines — the canonical map: discipline -> adapter slot, verifier, gate kind.

The Branch-Train-MiX / S-LoRA realisation of Sophia's council (``agent/swarm_router.py`` already
carries ``adapter_id`` "V3 (Branch-Train-MiX)"). A council seat is **one discipline served as a LoRA
adapter on a shared 3B base**, routed to per task, and admitted to the shared answer only if it
clears **its discipline's verifier** (the trust boundary, ``agent/swarm_trust_boundary.py``).

The honest core is the **gate kind** — what a seat can actually verify:

  * ``standalone`` — a deterministic validator gates ANY answer (mathematics, chemistry, biology).
  * ``reference``  — the verifier needs a gold/test (physics, coding); with no reference it ABSTAINS
                     (fail-closed) — it cannot verify a free-form claim it has no oracle for.
  * ``provenance`` — no truth oracle exists, so the seat is gated for SOURCE DISCIPLINE / attribution
                     (``agent.gate``): it reduces fabrication, it does not certify correctness.

A discipline with no machine verifier is NOT a verified expert — it is a provenance-gated one. That
distinction is the council's honesty: religion/history are PROTECTED (fixed floor, never RL-tuned).

Routing here is deterministic, lexical v1 (the same honesty bound as ``swarm_router``): it
generalises over phrasing, not deep meaning; the trained router overrides ``route`` later.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Discipline:
    id: str
    name: str
    gate: str                       # binding key in GATE_BINDINGS
    gate_kind: str                  # "standalone" | "reference" | "provenance"
    verifier_ref: str               # the verifier module/function (transparency)
    keywords: "tuple[str, ...]"
    tools: "frozenset[str]" = frozenset()
    protected: bool = False         # PROTECTED domains: fixed floor, never RL-optimised

    @property
    def adapter_slot(self) -> str:
        """The LoRA adapter id an S-LoRA / adapter-registry deployment would bind for this seat."""
        return f"sophia-{self.id}-3b"


def _d(id, name, gate, gate_kind, ref, keywords, tools=(), protected=False) -> Discipline:
    return Discipline(id, name, gate, gate_kind, ref, tuple(keywords), frozenset(tools), protected)


# --- the council ------------------------------------------------------------------------------
DISCIPLINES: "dict[str, Discipline]" = {d.id: d for d in [
    # STEM — machine-verifiable
    _d("mathematics", "Mathematics", "standalone:math", "standalone", "agent.verifiers.math_sound",
       ("prove", "theorem", "integral", "equation", "derivative", "probability", "algebra", "calculate", "=")),
    _d("physics", "Physics", "reference:physics", "reference", "agent.physics_verifier.verify",
       ("force", "velocity", "energy", "momentum", "quantum", "thermodynamic", "voltage", "newton", "joule")),
    _d("chemistry", "Chemistry", "standalone:chem", "standalone", "agent.chemistry_verifier.chemistry_sound",
       ("reaction", "molecule", "compound", "acid", "base", "oxidation", "stoichiometr", "->", "mole")),
    _d("biology", "Biology", "standalone:bio", "standalone", "agent.biology_verifier.biology_sound",
       ("dna", "rna", "protein", "cell", "gene", "enzyme", "organism", "sequence", "codon", "evolution")),
    _d("coding", "Software / Coding", "reference:code", "reference", "agent.code_verifier.verify",
       ("function", "code", "bug", "algorithm", "python", "compile", "unit test", "refactor", "api")),
    _d("engineering", "Engineering", "standalone:math", "standalone", "agent.verifiers.math_sound",
       ("circuit", "stress", "load", "design spec", "tolerance", "signal", "structural", "torque", "cad")),
    _d("statistics", "Statistics / Data science", "standalone:math", "standalone", "agent.verifiers.math_sound",
       ("mean", "variance", "p-value", "regression", "distribution", "confidence interval", "sample", "bayes")),
    _d("medicine", "Medicine", "provenance", "provenance", "agent.medical_faithfulness",
       ("diagnosis", "symptom", "treatment", "dose", "patient", "clinical", "disease", "drug", "therapy")),

    # Humanities — provenance-gated
    _d("philosophy", "Philosophy", "provenance", "provenance", "agent.gate (attribution)",
       ("ethics", "metaphysics", "epistemolog", "virtue", "kant", "aristotle", "argument", "meaning")),
    _d("history", "History", "provenance", "provenance", "agent.gate (temporal+attribution)",
       ("century", "empire", "war", "revolution", "ancient", "dynasty", "historical", "BCE", "CE"), protected=True),
    _d("religion", "Religion", "provenance", "provenance", "agent.gate (council)",
       ("scripture", "theolog", "buddhis", "christian", "islam", "daoist", "confucian", "sacred", "prophet"), protected=True),
    _d("linguistics", "Linguistics", "provenance", "provenance", "agent.gate (attribution)",
       ("grammar", "phoneme", "syntax", "etymolog", "morpholog", "language family", "dialect", "semantics")),

    # Social sciences — provenance-gated
    _d("psychology", "Psychology", "provenance", "provenance", "agent.gate (psychology_concepts)",
       ("cognition", "behavior", "emotion", "memory", "freud", "bias", "personality", "therapy", "perception")),
    _d("sociology", "Sociology / Social", "provenance", "provenance", "agent.gate (attribution)",
       ("society", "social", "class", "norm", "institution", "inequality", "community", "demograph", "culture")),
    _d("political_science", "Political science", "provenance", "provenance", "agent.gate (attribution)",
       ("government", "policy", "election", "democracy", "sovereign", "geopolit", "constitution", "vote", "regime")),
    _d("economics", "Economics", "provenance", "provenance", "agent.gate (numeric+attribution)",
       ("gdp", "inflation", "market", "supply", "demand", "fiscal", "monetary", "trade", "macroecon")),
    _d("finance", "Finance", "provenance", "provenance", "agent.gate (numeric+attribution)",
       ("portfolio", "valuation", "interest rate", "equity", "bond", "risk", "cash flow", "discount", "npv")),

    # Applied / professional — provenance-gated (law's citation existence is gated by agent.gate)
    _d("law", "Law", "provenance", "provenance", "agent.verifiers.legal_citation_exists",
       ("statute", "case", "court", "liability", "contract", "jurisdiction", "ruling", "v.", "plaintiff")),
    _d("business", "Business / Management", "provenance", "provenance", "agent.gate (attribution)",
       ("strategy", "management", "operations", "marketing", "supply chain", "kpi", "stakeholder", "org")),
    _d("education", "Education", "provenance", "provenance", "agent.gate (attribution)",
       ("pedagog", "curriculum", "learning outcome", "assessment", "classroom", "student", "teaching")),
    _d("environment", "Environment / Climate", "provenance", "provenance", "agent.gate (numeric+attribution)",
       ("climate", "emission", "ecosystem", "carbon", "biodiversity", "pollution", "sustainab", "renewable")),
]}

GENERAL = _d("general", "General", "provenance", "provenance", "agent.gate", ())


def disciplines() -> "list[Discipline]":
    return list(DISCIPLINES.values())


def get(discipline_id: str) -> Discipline:
    return DISCIPLINES.get(discipline_id, GENERAL)


def route(task: str) -> Discipline:
    """Deterministic lexical routing: the discipline whose keywords best match the task.
    No match -> the general seat. (Lexical v1; a trained router overrides this.)"""
    t = (task or "").lower()
    best, best_score = GENERAL, 0
    for d in DISCIPLINES.values():
        score = sum(1 for kw in d.keywords if kw in t)
        # the discipline's own id / name is a strong explicit cue ("in philosophy, ...")
        if d.id.replace("_", " ") in t or d.name.split()[0].lower() in t:
            score += 2
        if score > best_score:
            best, best_score = d, score
    return best


# --- verifier dispatch ------------------------------------------------------------------------
def _math(answer, _question, _gold):
    from agent.verifiers import math_sound
    r = math_sound()(answer, None, {})
    return bool(r["passed"]), list(r.get("reasons") or [])


def _chem(answer, _question, _gold):
    from agent.chemistry_verifier import chemistry_sound
    r = chemistry_sound()(answer, None, {})
    return bool(r["passed"]), list(r.get("reasons") or [])


def _bio(answer, _question, _gold):
    from agent.biology_verifier import biology_sound
    r = biology_sound()(answer, None, {})
    return bool(r["passed"]), list(r.get("reasons") or [])


def _provenance(answer, question, _gold):
    from agent.gate import check_response
    r = check_response(answer, mode="advisor", question=question or answer, route_claims=True)
    v = list(r.get("violations") or [])
    return (len(v) == 0), v


def _physics_ref(answer, _question, gold):
    if not gold:
        return None  # reference gate with no oracle -> abstain
    from agent.physics_verifier import verify
    r = verify(answer, gold)
    return (r.get("verdict") == "accepted"), list(r.get("reasons") or [])


def _code_ref(answer, _question, gold):
    if not gold:
        return None
    from agent.code_verifier import verify
    r = verify(answer, gold)  # gold is the hidden test_code
    return (r.get("verdict") == "accepted"), list(r.get("reasons") or [])


GATE_BINDINGS = {
    "standalone:math": _math, "standalone:chem": _chem, "standalone:bio": _bio,
    "provenance": _provenance, "reference:physics": _physics_ref, "reference:code": _code_ref,
}


def verify(discipline_id: str, answer: str, *, question: "str | None" = None,
           gold: "str | None" = None) -> dict:
    """Gate one seat's answer by ITS discipline verifier. A reference-gated seat with no gold
    ABSTAINS (passed False, abstained True) — fail-closed, never a guess."""
    d = get(discipline_id)
    res = GATE_BINDINGS[d.gate](answer, question, gold)
    if res is None:
        return {"discipline": d.id, "gate_kind": d.gate_kind, "passed": False, "abstained": True,
                "reasons": ["reference-gated seat: no gold/test supplied -> abstain"],
                "protected": d.protected}
    passed, reasons = res
    return {"discipline": d.id, "gate_kind": d.gate_kind, "passed": bool(passed),
            "abstained": False, "reasons": reasons, "protected": d.protected}


def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}

    checks["routes_math"] = route("Prove the integral of x equals x^2/2").id == "mathematics"
    checks["routes_biology"] = route("What does this DNA gene sequence encode?").id == "biology"
    checks["routes_law"] = route("Is this contract clause enforceable in court?").id == "law"
    checks["routes_finance"] = route("Compute the NPV of these cash flows at a 5% discount").id in ("finance", "mathematics")
    checks["unknown_routes_general"] = route("hello there").id == "general"

    # standalone gate: chemistry catches an unbalanced equation
    bad_chem = verify("chemistry", "The reaction is H2 + O2 -> H2O.")
    checks["chem_catches_unbalanced"] = (not bad_chem["passed"]) and bad_chem["gate_kind"] == "standalone"
    good_chem = verify("chemistry", "Balanced: 2 H2 + O2 -> 2 H2O.")
    checks["chem_passes_balanced"] = good_chem["passed"]

    # provenance gate: a forbidden attribution is caught
    bad_prov = verify("philosophy", "Confucius wrote the Dao De Jing.",
                      question="Did Confucius write the Dao De Jing?")
    checks["provenance_catches_merge"] = not bad_prov["passed"]

    # reference gate abstains without a gold
    ab = verify("coding", "def f(): return 1")
    checks["reference_abstains_without_gold"] = ab["abstained"] is True and not ab["passed"]

    # protected flag surfaces
    checks["history_protected"] = get("history").protected and get("religion").protected
    checks["adapter_slot"] = get("biology").adapter_slot == "sophia-biology-3b"
    checks["coverage"] = len(DISCIPLINES) >= 20

    ok = all(checks.values())
    return ok, {"checks": checks, "nDisciplines": len(DISCIPLINES)}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    ok, detail = offline_invariants()
    print("Council registry invariants:", "PASS" if ok else "FAIL", f"({detail['nDisciplines']} disciplines)")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)

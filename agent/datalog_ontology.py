# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Datalog concept-edge verifier — the symbolic half of the ontology gate.

This is the concept-TBox analogue of :mod:`agent.datalog_provenance`. The
*decision rule* for a proposed ontology edge — "abstain on an unscoped /
unsourced / identity cross-tradition claim; reject a disjointness violation;
admit only an intra-tradition or scoped+sourced edge" — is expressed as a
Datalog program over the closed-world engine (:mod:`agent.datalog_engine`); the
fact extraction (which traditions the endpoints belong to, whether a scope /
source exists) stays in Python and feeds ground facts to the engine.

The auditable rules (verified to run on the engine)::

    identityEdge(E) :- etype(E, sameAs).            # + equivalentClass / exactMatch / subClassOf
    scoped(E)       :- scopeOf(E, _R).
    sourced(E)      :- source(E, _S).
    abstain(E) :- crossTradition(E), identityEdge(E).   # Halpin: never blanket identity across vocab
    abstain(E) :- crossTradition(E), not scoped(E).     # the report's core rule
    abstain(E) :- crossTradition(E), not sourced(E).    # provenance-first
    violation(E) :- equates(E,A,B), classOf(A,Ca), classOf(B,Cb), disjoint(Ca,Cb).
    admit(E) :- edge(E), not abstain(E), not violation(E).

``crossTradition`` is computed in Python during extraction (the engine has no
inequality), exactly as the provenance port computes its lexical facts in Python.
This lets "wu wei resembles apatheia (scoped+sourced scopedAnalogy)" **admit**
while "wu wei ⊑ apatheia / ren ≡ agape" **abstain**.

See docs/11-Platform/Ontology-Claim-Boundary.md: a derived ``admit`` is NOT a
truth verdict — it only means the edge cleared the structural discipline; an
unverifiable cross-tradition truth claim can still only be quarantined.
"""
from __future__ import annotations

from typing import Any

from agent.datalog_engine import A, Atom, Program, Rule, Var

# Edge types that assert IDENTITY / subsumption (vs a scoped analogy).
_IDENTITY_EDGE_TYPES = ("sameAs", "equivalentClass", "exactMatch", "subClassOf")


def build_ontology_program() -> Program:
    """The pure Datalog kernel deriving ``abstain`` / ``violation`` / ``admit``.

    Returned unsolved; callers add the ground facts for the edges under review and
    call ``solve()``. Keeping the rules here (not inline) means the admission
    policy is auditable in one place.
    """
    prog = Program()
    for t in _IDENTITY_EDGE_TYPES:
        prog.rule(Rule.clause(A("identityEdge", Var("E")), A("etype", Var("E"), t)))
    prog.rule(Rule.clause(A("scoped", Var("E")), A("scopeOf", Var("E"), Var("R"))))
    prog.rule(Rule.clause(A("sourced", Var("E")), A("source", Var("E"), Var("S"))))
    # abstain: any cross-tradition identity, or a cross-tradition edge lacking a
    # scope or a source.
    prog.rule(Rule.clause(A("abstain", Var("E")), A("crossTradition", Var("E")), A("identityEdge", Var("E"))))
    prog.rule(Rule.clause(A("abstain", Var("E")), A("crossTradition", Var("E")), A("scoped", Var("E")).neg()))
    prog.rule(Rule.clause(A("abstain", Var("E")), A("crossTradition", Var("E")), A("sourced", Var("E")).neg()))
    # violation: the edge equates two concepts whose classes are declared disjoint.
    prog.rule(
        Rule.clause(
            A("violation", Var("E")),
            A("equates", Var("E"), Var("Ca"), Var("Cb")),
            A("classOf", Var("Ca"), Var("Ka")),
            A("classOf", Var("Cb"), Var("Kb")),
            A("disjoint", Var("Ka"), Var("Kb")),
        )
    )
    # admit: an edge that is neither abstained nor a violation.
    prog.rule(
        Rule.clause(
            A("admit", Var("E")),
            A("edge", Var("E")),
            A("abstain", Var("E")).neg(),
            A("violation", Var("E")).neg(),
        )
    )
    return prog


def _edge_id(edge: dict, idx: int) -> str:
    eid = edge.get("id")
    if eid:
        return str(eid)
    return f"e{idx}:{edge.get('subject')}|{edge.get('edgeType')}|{edge.get('object')}"


def _tradition_of(concept: str, declared, lexicon: dict) -> "str | None":
    if declared:
        return str(declared).lower()
    if lexicon and concept and str(concept).lower() in lexicon:
        return lexicon[str(concept).lower()]
    return None


def extract_facts(
    edges: list[dict], *, dnm: "dict | None" = None, lexicon: "dict | None" = None
) -> tuple[Program, dict]:
    """Extract ground facts for a list of ``ontology_edge`` dicts.

    ``dnm`` maps tradition -> set of do-not-merge traditions (-> ``disjoint``
    class facts); ``lexicon`` maps concept term -> tradition (to fill an absent
    ``subjectTradition`` / ``objectTradition``). Returns ``(program, meta)``.
    """
    if dnm is None or lexicon is None:
        from agent.verifiers import _load_concept_traditions, _load_dnm_by_tradition

        dnm = _load_dnm_by_tradition() if dnm is None else dnm
        lexicon = _load_concept_traditions() if lexicon is None else lexicon

    prog = build_ontology_program()
    meta: dict[str, Any] = {"edges": {}, "traditions": {}}

    # disjoint(Ka, Kb): symmetric, from the do-not-merge map (traditions == classes).
    for ta, forbidden in (dnm or {}).items():
        for tb in forbidden:
            prog.fact(A("disjoint", ta, tb))
            prog.fact(A("disjoint", tb, ta))

    for idx, edge in enumerate(edges):
        eid = _edge_id(edge, idx)
        subj = str(edge.get("subject") or "")
        obj = str(edge.get("object") or "")
        etype = str(edge.get("edgeType") or "")
        ta = _tradition_of(subj, edge.get("subjectTradition"), lexicon or {})
        tb = _tradition_of(obj, edge.get("objectTradition"), lexicon or {})

        prog.fact(A("edge", eid))
        if etype != "disjointWith":
            # "equates" is internal pair fact for disjoint-class violation detection
            # (used by identity + scoped claims). disjointWith decls populate "disjoint"
            # facts via DNM map; they do not emit equates.
            prog.fact(A("equates", eid, subj, obj))
        if etype:
            prog.fact(A("etype", eid, etype))
        if ta:
            prog.fact(A("classOf", subj, ta))
        if tb:
            prog.fact(A("classOf", obj, tb))
        if edge.get("scope"):
            prog.fact(A("scopeOf", eid, str(edge.get("scope"))))
        for s in edge.get("sources") or []:
            prog.fact(A("source", eid, str(s)))
        # crossTradition: computed here (the engine has no inequality).
        # Conservative for identity claims: unknown traditions => treat as cross (abstain path)
        # unless both traditions are known and identical.
        cross_trad = bool(ta and tb and ta != tb)
        if not cross_trad and (not ta or not tb) and etype in _IDENTITY_EDGE_TYPES:
            cross_trad = True
        if cross_trad:
            prog.fact(A("crossTradition", eid))
        meta["edges"][eid] = {"subject": subj, "object": obj, "edgeType": etype,
                              "subjectTradition": ta, "objectTradition": tb}
    return prog, meta


def classify_edges(
    edges: list[dict], *, dnm: "dict | None" = None, lexicon: "dict | None" = None
) -> dict:
    """Return ``{edge_id: verdict}`` where verdict ∈ {admit, abstain, violation}.

    A ``violation`` (disjointness) dominates ``abstain``, which dominates
    ``admit`` (an edge cannot be both admitted and abstained — the rules are
    mutually exclusive by construction, but we apply the precedence defensively).
    """
    prog, meta = extract_facts(edges, dnm=dnm, lexicon=lexicon)
    sol = prog.solve()
    derived = {pred: {a.args[0] for a in sol if a.pred == pred and a.args} for pred in ("admit", "abstain", "violation")}
    out: dict[str, str] = {}
    for eid in meta["edges"]:
        if eid in derived["violation"]:
            out[eid] = "violation"
        elif eid in derived["abstain"]:
            out[eid] = "abstain"
        elif eid in derived["admit"]:
            out[eid] = "admit"
        else:  # fail-closed: an edge we could not classify is quarantined
            out[eid] = "abstain"
    return out


def check_edge(edge: dict, *, dnm: "dict | None" = None, lexicon: "dict | None" = None) -> dict:
    """Classify a single edge. Returns ``{verdict, edgeId, detail}`` where verdict
    ∈ {admit, abstain, violation}. ``abstain`` maps to a quarantine (the only
    honest verdict for an unverifiable cross-tradition claim)."""
    prog, meta = extract_facts([edge], dnm=dnm, lexicon=lexicon)
    sol = prog.solve()
    (eid,) = list(meta["edges"])
    derived = {pred: {a.args[0] for a in sol if a.pred == pred and a.args} for pred in ("admit", "abstain", "violation")}
    if eid in derived["violation"]:
        verdict = "violation"
    elif eid in derived["abstain"]:
        verdict = "abstain"
    elif eid in derived["admit"]:
        verdict = "admit"
    else:
        verdict = "abstain"
    return {"verdict": verdict, "edgeId": eid, "detail": meta["edges"][eid]}


__all__ = ["Atom", "build_ontology_program", "extract_facts", "classify_edges", "check_edge"]

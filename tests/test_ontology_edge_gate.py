#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Falsification battery for the ontology concept-edge gate (P2). Offline.

The keystone the report found missing: a working concept-merge gate fed by the
doNotMergeWith / cross-tradition channel that the attribution gate never sees.

F1  surface gate fires on unscoped cross-tradition identity ("ren is identical to agape")
F2  surface gate passes a scoped analogy ("wu wei resembles apatheia ...")
F3  symbolic (Datalog) gate abstains on cross-tradition identity, admits scoped+sourced
SEIB structural-violation recall on disjoint pairs; no admit of an identity edge

See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.datalog_ontology import check_edge, classify_edges  # noqa: E402
from agent.guarded import check_claim  # noqa: E402
from agent.verifiers import ontology_edge_faithful  # noqa: E402


# --- F1: surface gate fires on unscoped cross-tradition identity ------------- #
def test_f1_surface_fires_on_cross_tradition_identity() -> None:
    for text in (
        "Ren is identical to agape.",
        "Wu wei is the same as apatheia.",
        "Ren is just agape.",
        "Ren equates to agape.",
        "Ren ≡ agape.",
    ):
        out = check_claim(text)
        assert out["passed"] is False, text
        assert out["violations"], text


# --- F2: scoped analogy / contrast is the admissible form (passes) ----------- #
def test_f2_scoped_analogy_passes() -> None:
    for text in (
        "Wu wei resembles apatheia with respect to effortless non-attached response.",
        "Ren is loosely analogous to agape, but they differ in their grounding.",
        "Ren is not agape.",
        "Wu wei and apatheia are distinct, though they invite comparison.",
    ):
        out = check_claim(text)
        assert out["passed"] is True, text
        assert out["violations"] == [], text


# --- F2b: the gate does not disturb the attribution / benign paths ----------- #
def test_f2b_attribution_and_benign_unchanged() -> None:
    assert check_claim("Confucius wrote the Dao De Jing.")["passed"] is False  # attribution still fires
    assert check_claim("Confucius did not write the Dao De Jing.")["passed"] is True
    assert check_claim("The library opens at nine in the morning.")["passed"] is True


# --- F3: symbolic gate — abstain on identity, admit only scoped+sourced ------ #
def test_f3_datalog_abstains_on_cross_tradition_identity() -> None:
    for etype in ("sameAs", "equivalentClass", "exactMatch", "subClassOf"):
        edge = {"subject": "ren", "object": "agape", "edgeType": etype,
                "subjectTradition": "confucian", "objectTradition": "christianity",
                "sources": ["x"], "scope": "love"}
        assert check_edge(edge)["verdict"] == "abstain", etype


def test_f3_datalog_admits_scoped_sourced_analogy() -> None:
    edge = {"subject": "wu wei", "object": "apatheia", "edgeType": "scopedAnalogy",
            "subjectTradition": "daoist", "objectTradition": "stoic",
            "sources": ["Graham 1989"], "scope": "effortless non-attached response"}
    assert check_edge(edge)["verdict"] == "admit"


def test_f3_datalog_abstains_when_scope_or_source_missing() -> None:
    no_scope = {"subject": "wu wei", "object": "apatheia", "edgeType": "scopedAnalogy",
                "subjectTradition": "daoist", "objectTradition": "stoic", "sources": ["x"]}
    no_source = {"subject": "wu wei", "object": "apatheia", "edgeType": "scopedAnalogy",
                 "subjectTradition": "daoist", "objectTradition": "stoic", "scope": "effortless"}
    assert check_edge(no_scope)["verdict"] == "abstain"
    assert check_edge(no_source)["verdict"] == "abstain"


def test_f3_datalog_admits_intra_tradition_subsumption() -> None:
    edge = {"subject": "ren", "object": "humaneness", "edgeType": "subClassOf",
            "subjectTradition": "confucian", "objectTradition": "confucian", "sources": ["x"]}
    assert check_edge(edge)["verdict"] == "admit"


# --- SEIB: structural-violation recall + no identity-edge admission ---------- #
def test_seib_disjointness_violation_detected() -> None:
    # confucian_ritual <-> christianity is an explicit doNotMergeWith pair.
    edge = {"subject": "ancestor_veneration", "object": "the_eucharist", "edgeType": "scopedAnalogy",
            "subjectTradition": "confucian_ritual", "objectTradition": "christianity",
            "sources": ["x"], "scope": "ritual"}
    assert check_edge(edge)["verdict"] == "violation"


def test_seib_no_cross_tradition_identity_ever_admitted() -> None:
    edges = [
        {"id": f"id{i}", "subject": "ren", "object": "agape", "edgeType": et,
         "subjectTradition": "confucian", "objectTradition": "christianity",
         "sources": ["x"], "scope": "love"}
        for i, et in enumerate(("sameAs", "equivalentClass", "exactMatch", "subClassOf"))
    ]
    verdicts = classify_edges(edges)
    assert "admit" not in set(verdicts.values()), verdicts


def main() -> int:
    test_f1_surface_fires_on_cross_tradition_identity()
    test_f2_scoped_analogy_passes()
    test_f2b_attribution_and_benign_unchanged()
    test_f3_datalog_abstains_on_cross_tradition_identity()
    test_f3_datalog_admits_scoped_sourced_analogy()
    test_f3_datalog_abstains_when_scope_or_source_missing()
    test_f3_datalog_admits_intra_tradition_subsumption()
    test_seib_disjointness_violation_detected()
    test_seib_no_cross_tradition_identity_ever_admitted()
    print("test_ontology_edge_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

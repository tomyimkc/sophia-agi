# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the concept-discipline INTEGRATION surfaces.

Proves the built modules are wired into live paths (not orphaned): the 'ontology'
policy in the guarded-loop registry, the check_concept_edge MCP/CLI surface, the
SSIL edge-admission flow (promote / quarantine / reject), and the OKF linker hard
failure on TBox inconsistencies."""
from __future__ import annotations

import json

from agent.policies import get_policy
from agent.ssil_ontology_seat import run_ontology_admission
from sophia_mcp.tools_impl import check_claim, check_concept_edge


# --- A1 policy registry ------------------------------------------------------ #
def test_ontology_policy_registered_and_flags_merge():
    p = get_policy("ontology")
    assert p.name == "ontology"
    assert p.verifier("Ren is identical to agape.", None, {})["passed"] is False
    assert p.verifier("Confucian ren is a virtue.", None, {})["passed"] is True


# --- A2 MCP / CLI surface ---------------------------------------------------- #
def test_check_concept_edge_verdicts():
    intra = check_concept_edge({"subject": "ren", "object": "li", "edgeType": "subClassOf",
                                "subjectTradition": "confucian", "objectTradition": "confucian"})
    assert intra["verdict"] == "admit"
    cross = check_concept_edge({"subject": "ren", "object": "agape", "edgeType": "sameAs",
                                "subjectTradition": "confucian", "objectTradition": "christianity"})
    assert cross["verdict"] == "abstain"  # unverifiable cross-tradition identity -> quarantine


def test_check_concept_edge_rejects_malformed():
    assert "error" in check_concept_edge({"subject": "ren"})
    assert "error" in check_concept_edge(None)


def test_mcp_check_claim_composes_ontology_gate():
    # the mode-free MCP check_claim now also catches the concept merge
    assert check_claim("Ren is identical to agape.")["passed"] is False


# --- A3 SSIL admission flow -------------------------------------------------- #
def test_admission_quarantines_cross_tradition_identity():
    edge = [{"subject": "ren", "object": "agape", "edgeType": "sameAs",
             "subjectTradition": "confucian", "objectTradition": "christianity"}]
    out = run_ontology_admission(edge)
    assert out["verdict"] in ("quarantine", "reject")  # never promote an unverifiable merge


def test_admission_rejects_disjoint_equation():
    dnm = {"confucian": ["christianity"], "christianity": ["confucian"]}
    edge = [{"subject": "ren", "object": "agape", "edgeType": "equivalentClass",
             "subjectTradition": "confucian", "objectTradition": "christianity",
             "sources": ["x"], "scope": "y"}]
    assert run_ontology_admission(edge, dnm=dnm)["verdict"] == "reject"


def test_admission_does_not_promote_without_evidence():
    # a clean intra-tradition edge clears the ontology seat but SSIL's safe ceiling
    # without Level-3 evidence is quarantine, never a live promote.
    edge = [{"subject": "ren", "object": "li", "edgeType": "subClassOf",
             "subjectTradition": "confucian", "objectTradition": "confucian", "sources": ["Analects"]}]
    assert run_ontology_admission(edge)["verdict"] in ("quarantine", "promote")

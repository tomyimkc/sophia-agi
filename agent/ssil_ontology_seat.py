# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""SSIL ontology seat (G_ontology) — a veto-only concept-edge gate.

Pluggable into :func:`agent.ssil.run_ssil` via ``extra_gates`` with zero
orchestrator change, shaped like :func:`agent.ssil_gates_ext.g3_capability_gate`.
It returns the standard ``{verdict, reasons, metrics}`` dict so the fail-closed
aggregator (``overall = min(verdicts, key=_PRECEDENCE)``) treats it like any seat.

Verdict policy (the only honest one until a ground-truth channel exists; see
docs/11-Platform/Ontology-Claim-Boundary.md):

  - any edge is a disjointness ``violation``           -> **reject** (veto)
  - else any edge is ``abstain`` (unverifiable cross-tradition / unscoped /
    unsourced)                                          -> **quarantine**
  - else every edge ``admit`` (intra-tradition or
    scoped+sourced)                                     -> **promote**

VETO-ONLY: the seat can block a fabricated/contradictory edge today, and it can
*admit* an edge that clears structural discipline, but it can never *admit a
cross-tradition truth claim as true* — that requires an independent ground-truth
channel the repo does not have, so such a claim is quarantined, never promoted.
"""
from __future__ import annotations

from typing import Any


def g_ontology_gate(edges: list[dict], *, dnm: "dict | None" = None, lexicon: "dict | None" = None) -> dict[str, Any]:
    """G_ontology seat — classify proposed concept edges and aggregate fail-closed."""
    from agent.datalog_ontology import classify_edges

    verdicts = classify_edges(edges, dnm=dnm, lexicon=lexicon) if edges else {}
    counts = {"admit": 0, "abstain": 0, "violation": 0}
    for v in verdicts.values():
        counts[v] = counts.get(v, 0) + 1

    if counts["violation"] > 0:
        verdict = "reject"
        reasons = [f"disjointness/identity violation in {counts['violation']} edge(s) — vetoed"]
    elif counts["abstain"] > 0:
        verdict = "quarantine"
        reasons = [f"{counts['abstain']} unverifiable cross-tradition/unscoped edge(s) — quarantined (no ground-truth channel)"]
    elif counts["admit"] > 0:
        verdict = "promote"
        reasons = [f"all {counts['admit']} edge(s) intra-tradition or scoped+sourced — cleared structural discipline"]
    else:  # no edges to judge: nothing to admit, nothing to block.
        verdict = "promote"
        reasons = ["no ontology edges in candidate"]

    return {
        "schema": "sophia.ssil_ontology_decision.v1", "gate": "G_ontology",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "verdict": verdict, "reasons": reasons,
        "metrics": {"counts": counts, "edgeVerdicts": verdicts, "vetoOnly": True},
    }


def _proposal_text(edges: list[dict]) -> str:
    parts = []
    for e in edges or []:
        parts.append(f"{e.get('subject')} {e.get('edgeType')} {e.get('object')}")
    return "propose ontology edges: " + "; ".join(parts) if parts else "propose ontology edges (none)"


def run_ontology_admission(
    edges: list[dict],
    *,
    dnm: "dict | None" = None,
    lexicon: "dict | None" = None,
    seed: int = 7,
) -> dict[str, Any]:
    """Run the fail-closed SSIL aggregator over proposed concept edges, with the
    ontology seat (G_ontology) + the public-standard value seat (G1) injected.

    This is the live admission flow: safety probes are all-pass (the candidate is a
    set of edges, not a code self-modification), so the decision is driven by
    G_ontology (veto bad edges / quarantine unverifiable / promote scoped+sourced)
    AND G1 (the proposal text clears the hard-floor public standard). Any seat's
    ``reject`` forces overall ``reject``; any ``quarantine`` forces ``quarantine``.
    """
    from agent.continual_plasticity import EvalMetric, UpdateCandidate
    from agent.corrigibility_gate import CorrigibilityCandidate, load_frozen_eval
    from agent.honeypot_gate import load_honeypots, select_active
    from agent.reward_isolation import AccessTrace
    from agent.ssil import SSILCandidate, run_ssil
    from agent.ssil_gates_ext import g1_moral_gate
    from agent.ssil_proposer import _full_surface

    corr_spec = load_frozen_eval()
    hp_spec = load_honeypots()
    active = select_active(hp_spec, seed=seed)
    all_corr = {s["id"]: True for s in corr_spec["scenarios"]}
    no_bait = {hp: False for hp in active}
    cid = "ontology_edge_admission"
    candidate = SSILCandidate(
        id=cid,
        access=AccessTrace(reads=frozenset({"data/concept_traditions.json"}),
                           writes=frozenset({"okf/graph.py"}), monitored=True,
                           proposed_text=_proposal_text(edges)),
        update=UpdateCandidate(id=cid, kind="ontology_edge",
                               verifier_artifacts=("datalog-classify",),
                               metrics=(EvalMetric("concept_discipline", 1.0, 1.0, protected=True),)),
        target_suite="concept_discipline",
        honeypots=dict(no_bait),
        corrigibility=CorrigibilityCandidate(id=cid, kind="ontology_edge", before=_full_surface(),
                                             after=_full_surface(), frozen_eval_results=dict(all_corr)),
        seed=seed,
        notes=_proposal_text(edges),
    )
    extra = {
        "G_ontology": g_ontology_gate(edges, dnm=dnm, lexicon=lexicon),
        "G1_moral": g1_moral_gate(_proposal_text(edges)),
    }
    return run_ssil(candidate, extra_gates=extra, surface=None,
                    honeypot_spec=hp_spec, corrigibility_eval=corr_spec)

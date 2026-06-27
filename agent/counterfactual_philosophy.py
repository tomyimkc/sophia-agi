# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Counterfactual philosophy generator (SANDBOX — never on the live path).

A fabrication generator by construction ("what would Aristotle conclude if he had
read the Dao De Jing?"). The provenance gate provably will NOT catch its output,
so safety here is **structural**, not behavioural:

  1. it writes ONLY to the bulk lattice (``okf.bulk_graph``), which forces
     ``bulkOnly=True`` — nothing it emits is a boundary (committed) node;
  2. every node/edge it emits carries ``sourceTier="counterfactual"`` +
     ``notPromotable=True`` so the taint is explicit and machine-readable;
  3. the promotion choke-point (``okf.promotion_loop.commit_approved_candidate``)
     hard-rejects ``sourceTier ∈ {counterfactual, synthetic}`` regardless of human
     approval (the information-flow invariant), and the SSIL provenance chain
     treats it as ``gateVerdict != promote`` so the taint propagates transitively.

Legitimate use (research §3.6): its output improves *reasoning* via RL (never SFT
on its content), and its failures become labelled negatives. The content is never
evidence. See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

from typing import Any

from okf.bulk_graph import BulkGraph, BulkHypothesis, BulkNode
from okf.graph import Graph

COUNTERFACTUAL_TIER = "counterfactual"
# The source tiers that may NEVER enter the boundary, regardless of approval.
NON_PROMOTABLE_TIERS = frozenset({"counterfactual", "synthetic"})


def counterfactual_meta(**over: Any) -> dict:
    """Frontmatter for a counterfactual node: tier-tagged and non-promotable."""
    meta = {
        "pageType": "concept",
        "sourceTier": COUNTERFACTUAL_TIER,
        "notPromotable": True,
        "bulkOnly": True,
        "authorConfidence": "none_extant",  # weakest rank: cannot launder downstream
    }
    meta.update(over)
    return meta


def generate_counterfactual(
    *,
    figure: str,
    source_concept: str,
    counterpart_concept: str,
    boundary: "Graph | None" = None,
    note: str = "",
) -> BulkGraph:
    """Emit a single counterfactual hypothesis into a fresh bulk lattice.

    Models "what would ``figure`` conclude about ``source_concept`` if exposed to
    ``counterpart_concept``" as a *quoted hypothesis in a separate world*, never an
    ABox edge. Returns the ``BulkGraph`` (in-memory, non-canonical).
    """
    bulk = BulkGraph(boundary=boundary if boundary is not None else Graph())
    node_id = f"cf_{figure}_{source_concept}_{counterpart_concept}".lower().replace(" ", "_")
    body = (
        f"COUNTERFACTUAL (non-evidence): a quoted hypothesis exploring how {figure} "
        f"might relate {source_concept} to {counterpart_concept}. This is fabricated "
        f"reasoning material for RL-on-reasoning only; it is never a sourced claim."
    )
    bulk.add_node(node_id, meta=counterfactual_meta(
        id=node_id, title=f"counterfactual: {figure} on {source_concept}~{counterpart_concept}",
    ), body=body)
    bulk.add_hypothesis(
        source=source_concept, edge="scopedAnalogy", target=counterpart_concept,
        note=note or f"counterfactual {figure}; sourceTier={COUNTERFACTUAL_TIER}; notPromotable",
    )
    return bulk


def is_promotable_tier(source_tier: "str | None") -> bool:
    """True iff a source tier may ever be promoted to the boundary."""
    return str(source_tier or "").lower() not in NON_PROMOTABLE_TIERS


def audit_counterfactual(bulk: BulkGraph) -> dict:
    """A machine-readable audit asserting every emitted node is tier-tagged and
    bulk-only (the structural invariant, checkable in CI)."""
    nodes: list[BulkNode] = list(bulk.nodes.values())
    hyps: list[BulkHypothesis] = list(bulk.hypotheses)
    all_tagged = all(n.meta.get("sourceTier") in NON_PROMOTABLE_TIERS for n in nodes)
    all_bulk_only = all(bool(n.bulkOnly) and bool(n.meta.get("bulkOnly")) for n in nodes)
    all_non_promotable = all(bool(n.meta.get("notPromotable")) for n in nodes)
    return {
        "schema": "sophia.counterfactual_audit.v1",
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "nodeCount": len(nodes), "hypothesisCount": len(hyps),
        "allTierTagged": all_tagged,
        "allBulkOnly": all_bulk_only,
        "allNonPromotable": all_non_promotable,
        "structurallyContained": all_tagged and all_bulk_only and all_non_promotable,
    }


__all__ = [
    "COUNTERFACTUAL_TIER", "NON_PROMOTABLE_TIERS", "counterfactual_meta",
    "generate_counterfactual", "is_promotable_tier", "audit_counterfactual",
]

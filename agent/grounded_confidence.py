# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Provenance-derived confidence for the graded answer/hedge/abstain router.

The graded router (`agent.graded_decision`) was wired into `grounded_answer` in L1, but its
*confidence input* still had to be supplied by the caller — a synthetic signal. This module
closes that gap with a **live, grounded** confidence: it reads the provenance of the routed
OKF page and its k-hop neighborhood (the same neighborhood retrieval already expands over)
and pools it into a single confidence with the existing Bayesian corroboration combiner.

Two real signals drive it, no model call:
  - **source quality** — each page's ``authorConfidence`` tier maps to a prior P(claim is
    well-grounded | a source of this tier). A ``consensus`` source is strong evidence; a
    ``legendary`` / ``disputed`` one is weak.
  - **corroboration** — independent neighboring sources that agree raise the pooled
    confidence (`agent.corroboration.corroborated_confidence`, log-odds). Sources sharing a
    ``tradition`` are one independence group, so a single tradition cannot double-count.

A ``contradicts`` edge in the neighborhood is treated as dissent (confidence < 0.5), so a
contested claim is pulled down rather than ignored. Deterministic, offline, dependency-free.

Honest bound: this scores *how well-sourced the routed page is*, not whether a specific
generated sentence is true — it is a calibrated prior for the router's downgrade decision,
not a fact-checker. The router stays fail-closed; this only ever lowers a weak answer.
"""

from __future__ import annotations

from agent.corroboration import Evidence, corroborated_confidence

#: P(claim well-grounded | a source of this authorConfidence tier). Ordered strong→weak.
AUTHOR_CONFIDENCE_PRIOR = {
    "consensus": 0.92,
    "attributed": 0.82,
    "compiled": 0.76,
    "layered": 0.70,
    "disputed": 0.45,          # contested authorship: weak/ambiguous support
    "legendary": 0.40,
    "anachronism_risk": 0.35,
    "none_extant": 0.30,
}
#: Used when a page declares no authorConfidence (unknown → mildly below neutral).
_DEFAULT_PRIOR = 0.55
#: A neighbor reached via a `contradicts` edge is dissent, not support.
_CONTRADICT_CONFIDENCE = 0.30


def _prior_for(meta: dict) -> float:
    return AUTHOR_CONFIDENCE_PRIOR.get(meta.get("authorConfidence"), _DEFAULT_PRIOR)


def corroboration_evidence_for(target: str, pages, *, hops: int = 1) -> "list[Evidence]":
    """Build corroboration evidence from the routed page + its k-hop OKF neighborhood.

    Each page contributes one Evidence whose confidence is its ``authorConfidence`` prior and
    whose independence group is its ``tradition`` (so same-tradition sources don't double-
    count). Neighbors reached only via a ``contradicts`` edge from the target count as dissent.
    Returns ``[]`` when the target is unknown, so the caller can fall back to no-signal.
    """
    from okf import build_graph  # noqa: PLC0415
    from agent.continual_qa_answer import neighborhood_ids  # noqa: PLC0415
    from okf.schema import as_list  # noqa: PLC0415

    page_list = list(pages)
    by_id = {p.id: p for p in page_list}
    if target not in by_id:
        return []
    graph = build_graph(page_list)
    ids = neighborhood_ids(graph, target, hops=hops)

    # Ids the target explicitly contradicts (resolved) — treated as dissent.
    from okf.graph import resolve  # noqa: PLC0415
    contra = set()
    tnode = graph.nodes.get(target)
    if tnode is not None:
        for raw in as_list(tnode["meta"].get("contradicts")):
            rid = resolve(graph, str(raw))
            if rid is not None:
                contra.add(rid)

    evidence: list[Evidence] = []
    for pid in ids:
        page = by_id.get(pid)
        if page is None:
            continue
        meta = page.meta
        conf = _CONTRADICT_CONFIDENCE if pid in contra else _prior_for(meta)
        group = str(meta.get("tradition") or pid)
        evidence.append(Evidence(source_id=pid, confidence=conf, independence_group=group))
    return evidence


def grounded_source_confidence(target: str, pages, *, hops: int = 1) -> "float | None":
    """Pooled P(routed page is well-grounded) over its neighborhood, or None if unknown.

    None (not 0.5) signals "no usable provenance signal" so the router can skip grading
    rather than treat absence of evidence as a neutral confidence.
    """
    evidence = corroboration_evidence_for(target, pages, hops=hops)
    if not evidence:
        return None
    return float(corroborated_confidence(evidence))


__all__ = [
    "AUTHOR_CONFIDENCE_PRIOR",
    "corroboration_evidence_for",
    "grounded_source_confidence",
]

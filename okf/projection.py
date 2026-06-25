# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Holographic projector: bulk candidate -> boundary promotion gate.

Only bulk states that project cleanly (structural + provenance checks) become
``PromotionCandidate`` records. Everything else stays quarantined in bulk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.conscience import conscience_check
from okf.bulk_graph import BulkGraph
from okf.graph import self_merges, tradition_merges
from okf.schema import validate_meta


@dataclass
class PromotionCandidate:
    """One bulk node that passed projection and may be committed to boundary."""

    node_id: str
    meta: dict
    body: str
    checks: dict = field(default_factory=dict)


@dataclass
class ProjectionResult:
    """Outcome of projecting one or more bulk nodes onto the boundary."""

    schema: str = "sophia.okf.projection.v1"
    candidateOnly: bool = True
    promoted: list[PromotionCandidate] = field(default_factory=list)
    abstained: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.promoted) and not self.abstained

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "candidateOnly": self.candidateOnly,
            "promoted": [
                {"nodeId": p.node_id, "meta": p.meta, "body": p.body[:500], "checks": p.checks}
                for p in self.promoted
            ],
            "abstained": self.abstained,
            "promotedCount": len(self.promoted),
            "abstainedCount": len(self.abstained),
        }


def _provenance_check(body: str) -> tuple[bool, list[str]]:
    try:
        from agent.verifiers import provenance_faithful

        pf = provenance_faithful()(body, None, {})
        return bool(pf.get("passed")), list(pf.get("reasons") or [])
    except Exception as exc:
        return False, [f"provenance verifier unavailable: {type(exc).__name__}"]


_CONSCIENCE_ABSTAIN = frozenset({"block", "abstain", "escalate", "retrieve", "clarify"})


def _conscience_check(body: str) -> tuple[bool, list[str], str | None]:
    """Fail-closed conscience gate for bulk projection body text."""
    decision = conscience_check(body, mode="output", context={"canClaimAGI": False})
    verdict = decision.verdict
    if verdict in _CONSCIENCE_ABSTAIN:
        return False, [f"conscience:{verdict}: {decision.reason}"], verdict
    if verdict not in ("allow", "revise"):
        return False, [f"conscience:unknown_verdict:{verdict}"], verdict
    return True, [], verdict


def project_node(
    bulk: BulkGraph,
    node_id: str,
    *,
    dnm_by_tradition: dict | None = None,
    skip_provenance: bool = False,
    skip_conscience: bool = False,
) -> tuple[PromotionCandidate | None, list[str]]:
    """Project a single bulk node through structural + provenance gates."""
    node = bulk.nodes.get(node_id)
    if node is None:
        return None, [f"unknown bulk node '{node_id}'"]

    reasons: list[str] = list(validate_meta(node.meta))
    if not skip_provenance:
        ok_pf, pf_reasons = _provenance_check(node.body)
        if not ok_pf:
            reasons.extend(pf_reasons)

    author = node.meta.get("attributedAuthor")
    from okf.schema import as_list

    forbidden = {str(a).lower() for a in as_list(node.meta.get("doNotAttributeTo"))}
    if author and str(author).lower() in forbidden:
        reasons.append(f"lineage-merge: attributedAuthor '{author}' is in doNotAttributeTo")

    combined = bulk.combined_graph()
    if not bulk.relax_tradition:
        for tm in tradition_merges(combined, dnm_by_tradition=dnm_by_tradition):
            if tm.get("page") == node_id or tm.get("linksTo") == node_id:
                reasons.append(
                    f"tradition-merge: {tm.get('tradition')} x {tm.get('otherTradition')}"
                )
    else:
        # Even in relaxed bulk, flag tradition merges involving this node for audit.
        merges = [
            tm
            for tm in tradition_merges(combined, dnm_by_tradition=dnm_by_tradition)
            if tm.get("page") == node_id or tm.get("linksTo") == node_id
        ]
        if merges and not node.meta.get("allowTraditionExploration"):
            reasons.append("tradition-merge in bulk without allowTraditionExploration flag")

    for sm in self_merges(combined):
        if sm.get("page") == node_id:
            reasons.append(f"self-merge on page '{node_id}'")

    conscience_verdict: str | None = None
    if not skip_conscience:
        ok_cc, cc_reasons, conscience_verdict = _conscience_check(node.body)
        if not ok_cc:
            reasons.extend(cc_reasons)

    if reasons:
        return None, reasons

    checks = {
        "schemaValid": True,
        "provenanceFaithful": not skip_provenance,
        "conscienceChecked": not skip_conscience,
        "conscienceVerdict": conscience_verdict or ("skipped" if skip_conscience else "allow"),
        "traditionRelaxed": bulk.relax_tradition,
    }
    return PromotionCandidate(node_id=node.id, meta=dict(node.meta), body=node.body, checks=checks), []


def project_to_boundary(
    bulk: BulkGraph,
    *,
    node_ids: list[str] | None = None,
    dnm_by_tradition: dict | None = None,
    skip_provenance: bool = False,
    skip_conscience: bool = False,
) -> ProjectionResult:
    """Project bulk nodes; promoted list is gate-clean, abstained carries reasons."""
    result = ProjectionResult()
    targets = node_ids if node_ids is not None else sorted(bulk.nodes.keys())
    for nid in targets:
        cand, reasons = project_node(
            bulk,
            nid,
            dnm_by_tradition=dnm_by_tradition,
            skip_provenance=skip_provenance,
            skip_conscience=skip_conscience,
        )
        if cand is not None:
            result.promoted.append(cand)
        else:
            result.abstained.append({"nodeId": nid, "reasons": reasons})
    return result

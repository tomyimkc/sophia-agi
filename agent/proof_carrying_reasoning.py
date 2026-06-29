# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Proof-carrying reasoning over the grounded OKF belief graph (VeriCoT-style).

Sophia's existing gates are strong on *single-claim* provenance (is the source
grounded? is its confidence laundered?) but a multi-hop answer can still slip a
**synthesis hallucination** through: each individual hop looks plausible, yet the
chain rests, somewhere, on a premise that is not actually grounded — an
"as everyone knows" leap, a citation to a page that does not exist, or a step that
quietly cites a conclusion it has not yet earned. A face-value reader accepts the
whole answer; the gate, looking hop-by-hop, misses the gap.

This module closes that gap by making a reasoning chain *carry its own proof*. It
follows the VeriCoT recipe (arXiv:2511.04662): autoformalize each step, tag every
premise by where its warrant comes from, and verify the chain end to end.

  * **Autoformalize.** Each step's conclusion and each grounded-fact premise are
    atomic claims in the ``agent.formal_verifier`` shape
    ``{"subject","predicate","object","negated"}``. The whole set is run through
    ``check_no_contradiction`` — a chain that asserts X and not-X *anywhere* is
    rejected, not silently synthesized.
  * **Tag premises by provenance.** A premise is one of:
      - ``grounded_fact`` — must be ``okf.is_grounded`` (resolved via the symbol
        layer) AND its page's effective (min-over-``derivesFrom``) confidence must
        clear a floor. An unresolvable / ungrounded / under-floor ref is *not* a
        warrant.
      - ``prior_step`` — must name an EARLIER step that already verified. A forward
        or self reference earns nothing (fail-closed).
      - ``commonsense`` — an explicit, *unverifiable* assumption. It marks the step
        assumption-bearing; unless commonsense is explicitly allowed, it makes the
        step unverifiable.
  * **Verify the chain in order.** A step is verified iff *every* premise is
    verified. If any step is unverifiable the whole answer **abstains** (it is not
    downgraded to a lower-confidence assertion); a contradiction **rejects** it.

The discipline is fail-closed throughout: an unverifiable premise never yields an
accepted answer, only ``abstain`` / ``rejected`` / withheld. Every emitted verdict
carries ``candidateOnly: True`` — this checks the *internal warrant structure* of a
chain, not the empirical truth of its conclusion. Pure standard library; the z3
backend is optional and inherited (fail-closed) from ``agent.formal_verifier``.

    from okf import build_graph
    from agent.proof_carrying_reasoning import verify_chain
    g = build_graph(pages)
    verify_chain(g, steps)["verdict"]   # "verified" | "abstain" | "rejected"
"""

from __future__ import annotations

from typing import Any

from okf import is_grounded, propagate_confidence
from okf.schema import confidence_rank
from agent.formal_verifier import check_no_contradiction
from agent.symbol_identity import canonical_id, stable_identity, version_tag

# Default confidence floor for a grounded-fact premise. A premise resting on a page
# whose effective (min-over-derivesFrom) rank is below this is treated as too weak to
# warrant a step — fail-closed, so a "legendary"/"none_extant" source cannot silently
# carry a multi-hop conclusion. ``compiled`` (rank 2) is the default admission floor.
DEFAULT_MIN_CONFIDENCE_RANK = confidence_rank("compiled")

SCHEMA = "sophia.proof_carrying_reasoning.v1"

# Premise provenance types.
GROUNDED_FACT = "grounded_fact"
PRIOR_STEP = "prior_step"
COMMONSENSE = "commonsense"


# --------------------------------------------------------------------------- #
# Atomic-claim helpers
# --------------------------------------------------------------------------- #
def _is_atomic(claim: Any) -> bool:
    """A claim is atomic iff it carries subject/predicate/object (negated optional)."""
    return (
        isinstance(claim, dict)
        and "subject" in claim
        and "predicate" in claim
        and "object" in claim
    )


def _normalize_atomic(claim: dict) -> dict:
    """Coerce a claim to the formal_verifier shape with an explicit ``negated``."""
    return {
        "subject": claim.get("subject"),
        "predicate": claim.get("predicate"),
        "object": claim.get("object"),
        "negated": bool(claim.get("negated", False)),
    }


# --------------------------------------------------------------------------- #
# Premise verification
# --------------------------------------------------------------------------- #
def _verify_grounded_fact(graph, premise: dict, min_confidence_rank: int) -> dict:
    """Verify a grounded_fact premise: it must resolve to a present, grounded page
    whose effective confidence clears the floor. Fail-closed on every gap."""
    ref = premise.get("ref")
    verdict = {
        "type": GROUNDED_FACT,
        "ref": ref,
        "verified": False,
        "resolvedId": None,
        "grounded": False,
        "effectiveConfidenceRank": None,
        "reasons": [],
    }
    if premise.get("claim") is not None and not _is_atomic(premise.get("claim")):
        verdict["reasons"].append("grounded_fact claim is not atomic")
        return verdict

    nid = canonical_id(graph, ref) if ref is not None else None
    if nid is None or nid not in graph.nodes:
        verdict["reasons"].append(f"ungrounded: ref {ref!r} resolves to no page")
        return verdict
    verdict["resolvedId"] = nid

    grounded = is_grounded(graph, nid)
    verdict["grounded"] = grounded
    if not grounded:
        verdict["reasons"].append(f"ungrounded: {nid} has lost its provenance ground")
        return verdict

    rank = propagate_confidence(graph).get(nid, 0)
    verdict["effectiveConfidenceRank"] = rank
    if rank < min_confidence_rank:
        verdict["reasons"].append(
            f"below floor: {nid} effective rank {rank} < {min_confidence_rank}"
        )
        return verdict

    verdict["verified"] = True
    verdict["reasons"].append(f"grounded fact {nid} clears confidence floor")
    return verdict


def _verify_prior_step(premise: dict, verified_prior_ids) -> dict:
    """Verify a prior_step premise: its ref must be an EARLIER, already-verified step.
    A forward or self reference is not a warrant (fail-closed)."""
    ref = premise.get("ref")
    verified = ref in set(verified_prior_ids)
    return {
        "type": PRIOR_STEP,
        "ref": ref,
        "verified": verified,
        "reasons": [
            f"prior step {ref!r} verified earlier"
            if verified
            else f"not an earlier verified step: {ref!r}"
        ],
    }


def _verify_commonsense(premise: dict, allow_commonsense: bool) -> dict:
    """A commonsense premise is an explicit, unverifiable assumption. It is only
    'verified' when commonsense is explicitly allowed — and even then it is flagged
    as an assumption rather than a grounded warrant."""
    text = premise.get("text")
    return {
        "type": COMMONSENSE,
        "text": text,
        "verified": bool(allow_commonsense),
        "assumption": True,
        "reasons": [
            "commonsense assumption admitted (flagged, not grounded)"
            if allow_commonsense
            else "commonsense premise is unverifiable (allow_commonsense=False)"
        ],
    }


# --------------------------------------------------------------------------- #
# Public API: per-step verification
# --------------------------------------------------------------------------- #
def verify_step(
    graph,
    step: dict,
    *,
    verified_prior_ids,
    min_confidence_rank: int = DEFAULT_MIN_CONFIDENCE_RANK,
    allow_commonsense: bool = False,
) -> dict:
    """Verify a single reasoning step against the graph and the verified-so-far set.

    A step is ``verified`` iff EVERY premise is verified:
      * a ``grounded_fact`` premise is grounded in the graph and above the floor;
      * a ``prior_step`` premise names an id in ``verified_prior_ids`` (earlier &
        verified);
      * a ``commonsense`` premise marks the step assumption-bearing and is only a
        warrant when ``allow_commonsense`` is True.

    Returns ``{"stepId","verified","premises","unverifiablePremises",
    "assumptionBearing","conclusion","candidateOnly"}``. Fail-closed: a malformed
    conclusion or premise, or any unverifiable premise, leaves ``verified`` False.
    """
    step_id = step.get("id")
    conclusion = step.get("conclusion")
    premises = step.get("premises") or []

    premise_verdicts: list[dict] = []
    reasons: list[str] = []
    assumption_bearing = False

    if not _is_atomic(conclusion):
        reasons.append("step conclusion is not an atomic claim")

    for premise in premises:
        ptype = premise.get("type") if isinstance(premise, dict) else None
        if ptype == GROUNDED_FACT:
            pv = _verify_grounded_fact(graph, premise, min_confidence_rank)
        elif ptype == PRIOR_STEP:
            pv = _verify_prior_step(premise, verified_prior_ids)
        elif ptype == COMMONSENSE:
            pv = _verify_commonsense(premise, allow_commonsense)
            assumption_bearing = True
        else:
            pv = {
                "type": ptype,
                "verified": False,
                "reasons": [f"unknown premise type: {ptype!r}"],
            }
        premise_verdicts.append(pv)

    unverifiable = [pv for pv in premise_verdicts if not pv.get("verified")]
    verified = (
        _is_atomic(conclusion)
        and not unverifiable
        and all(pv.get("verified") for pv in premise_verdicts)
    )

    return {
        "stepId": step_id,
        "verified": bool(verified),
        "conclusion": _normalize_atomic(conclusion) if _is_atomic(conclusion) else None,
        "premises": premise_verdicts,
        "unverifiablePremises": unverifiable,
        "assumptionBearing": assumption_bearing,
        "reasons": reasons,
        "candidateOnly": True,
    }


# --------------------------------------------------------------------------- #
# Autoformalization
# --------------------------------------------------------------------------- #
def autoformalize_claims(steps) -> "list[dict]":
    """Collect the atomic-claim set for a chain: every step conclusion plus every
    grounded-fact premise claim, normalized to the formal_verifier shape. This is
    the set fed to ``check_no_contradiction`` — the chain's formal footprint."""
    claims: list[dict] = []
    for step in steps:
        conclusion = step.get("conclusion")
        if _is_atomic(conclusion):
            claims.append(_normalize_atomic(conclusion))
        for premise in step.get("premises") or []:
            if isinstance(premise, dict) and premise.get("type") == GROUNDED_FACT:
                claim = premise.get("claim")
                if _is_atomic(claim):
                    claims.append(_normalize_atomic(claim))
    return claims


# --------------------------------------------------------------------------- #
# Public API: whole-chain verification
# --------------------------------------------------------------------------- #
def verify_chain(
    graph,
    steps,
    *,
    min_confidence_rank: int = DEFAULT_MIN_CONFIDENCE_RANK,
    allow_commonsense: bool = False,
) -> dict:
    """Verify a reasoning chain end to end and emit a proof-carrying result.

    Steps are verified IN ORDER, so a ``prior_step`` premise can cite only steps
    that already verified (a forward / self reference earns nothing). The whole
    autoformalized claim set is run through ``check_no_contradiction``.

    Verdict (fail-closed):
      * ``"rejected"`` — the chain asserts X and not-X somewhere (contradiction).
      * ``"abstain"``  — some step is unverifiable (ungrounded / under-floor /
        forward-cited prior / disallowed commonsense). The answer is withheld, NOT
        downgraded to a weaker assertion.
      * ``"verified"`` — every step verified and the claim set is consistent.

    Returns the ``sophia.proof_carrying_reasoning.v1`` record: ``verdict``,
    ``verifiedSteps``, ``abstainedSteps``, ``premiseChain`` (the verified premise
    lineage), ``contradiction`` (the formal_verifier result), ``stepResults``, and
    ``reasons``. Always ``candidateOnly: True``.
    """
    verified_prior_ids: list[str] = []
    step_results: list[dict] = []
    verified_steps: list[str] = []
    abstained_steps: list[str] = []
    premise_chain: list[dict] = []
    reasons: list[str] = []

    for step in steps:
        result = verify_step(
            graph,
            step,
            verified_prior_ids=verified_prior_ids,
            min_confidence_rank=min_confidence_rank,
            allow_commonsense=allow_commonsense,
        )
        step_results.append(result)
        sid = result["stepId"]
        if result["verified"]:
            verified_steps.append(sid)
            verified_prior_ids.append(sid)
            # Record the verified premise lineage for this step.
            for pv in result["premises"]:
                entry = {"stepId": sid, "premise": pv}
                premise_chain.append(entry)
        else:
            abstained_steps.append(sid)
            for pv in result["unverifiablePremises"]:
                for r in pv.get("reasons", []):
                    reasons.append(f"step {sid}: {r}")
            for r in result.get("reasons", []):
                reasons.append(f"step {sid}: {r}")

    # Autoformalize + solver-verify the whole claim set.
    claims = autoformalize_claims(steps)
    contradiction = check_no_contradiction(claims) if claims else None
    has_contradiction = bool(
        contradiction and contradiction.get("verdict") == "rejected"
    )

    if has_contradiction:
        verdict = "rejected"
        reasons.append("contradiction in autoformalized chain (asserts X and not-X)")
    elif abstained_steps:
        verdict = "abstain"
        reasons.append(
            f"{len(abstained_steps)} step(s) rest on an unverifiable premise; "
            "answer withheld (fail-closed)"
        )
    else:
        verdict = "verified"
        reasons.append("all steps verified and chain is internally consistent")

    return {
        "schema": SCHEMA,
        "candidateOnly": True,
        "verdict": verdict,
        "verifiedSteps": verified_steps,
        "abstainedSteps": abstained_steps,
        "premiseChain": premise_chain,
        "contradiction": contradiction,
        "stepResults": step_results,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# Public API: proof-carrying answer
# --------------------------------------------------------------------------- #
def _grounded_citations(graph, premise_chain) -> "list[dict]":
    """Stable citations for every grounded fact used in the verified premise chain.

    Each carries the ref, the resolved node id, and its ``stable_identity`` /
    ``version_tag`` from ``agent.symbol_identity`` — a fact's invariant address, so
    the citation survives supersession and retract/restore. Deduplicated by
    (stepId, resolvedId), order-preserving."""
    citations: list[dict] = []
    seen: set = set()
    for entry in premise_chain:
        pv = entry.get("premise", {})
        if pv.get("type") != GROUNDED_FACT or not pv.get("verified"):
            continue
        nid = pv.get("resolvedId")
        key = (entry.get("stepId"), nid)
        if nid is None or key in seen:
            continue
        seen.add(key)
        citations.append({
            "stepId": entry.get("stepId"),
            "ref": pv.get("ref"),
            "resolvedId": nid,
            "stableIdentity": stable_identity(graph, nid),
            "versionTag": version_tag(graph, nid),
            "effectiveConfidenceRank": pv.get("effectiveConfidenceRank"),
        })
    return citations


def proof_carrying_answer(
    graph,
    question: str,
    steps,
    *,
    min_confidence_rank: int = DEFAULT_MIN_CONFIDENCE_RANK,
    allow_commonsense: bool = False,
) -> dict:
    """Wrap ``verify_chain`` into an answer that carries its proof.

    On a ``"verified"`` verdict the answer is released *with* a citation list giving
    each grounded fact a stable identity / version tag (so the warrant is auditable
    and survives revision). On any other verdict the answer is **withheld** with the
    reasons why — never silently emitted. Always ``candidateOnly: True``.
    """
    chain = verify_chain(
        graph,
        steps,
        min_confidence_rank=min_confidence_rank,
        allow_commonsense=allow_commonsense,
    )
    verdict = chain["verdict"]
    out = {
        "schema": SCHEMA,
        "candidateOnly": True,
        "question": question,
        "verdict": verdict,
        "chain": chain,
    }
    if verdict == "verified":
        out["answerReleased"] = True
        out["citations"] = _grounded_citations(graph, chain["premiseChain"])
        out["assumptionBearing"] = any(
            r.get("assumptionBearing") for r in chain["stepResults"]
        )
    else:
        out["answerReleased"] = False
        out["answerWithheld"] = True
        out["why"] = list(chain["reasons"])
    return out


__all__ = [
    "DEFAULT_MIN_CONFIDENCE_RANK",
    "SCHEMA",
    "GROUNDED_FACT",
    "PRIOR_STEP",
    "COMMONSENSE",
    "autoformalize_claims",
    "verify_step",
    "verify_chain",
    "proof_carrying_answer",
]

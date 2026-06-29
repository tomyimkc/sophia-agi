# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic GRPO reward for RETRIEVAL-FAITHFUL reasoning (verifier-as-reward).

Extends ``provenance_bench.rl_reward`` from "don't assert a forbidden
attribution" to the full retrieve-then-reason contract a *knowledge-from-wiki*
reasoning core must satisfy: ground every knowledge claim in a RETRIEVED chunk,
defer to that chunk instead of the weights, never assert more confidence than
the source chain supports, and decide *when* to retrieve vs answer directly.

Like ``rl_reward``, this is a deterministic, bounded ``[-1, 1]`` reward driven
only by the verifier seam — NOT a model self-score. The one piece that needs a
model (the counterfactual citation-drop test) is run by the ROLLOUT harness,
not here: the harness regenerates each answer with a claim's supporting chunk
removed and records, per claim, whether the claim SURVIVED the ablation. This
module consumes that already-collected, machine-checkable trajectory, so the
reward computation itself stays deterministic and CI-testable offline (no
torch, no GPU) exactly as ``tests/test_rlvr.py`` requires.

Faithfulness, defined operationally: a knowledge claim is FAITHFUL iff it
DISAPPEARS when its supporting chunk is dropped (the model was genuinely using
retrieval). A claim that SURVIVES the drop leaked from the weights — true or
not, it violates the "truth outside the weights" thesis and is penalized. This
is the signal that separates a reasoning core over a wiki from a closet
memorizer, and it is the term ``rl_reward`` did not have.

Honest scope: this trains retrieval-grounded *behaviour* (ground / defer /
calibrate-confidence / decide-to-retrieve). It does not train world knowledge
(that lives in the wiki) and it does not by itself certify any capability — the
live GRPO uplift stays gated until a pre-registered, powered, multi-family run.
See ``agi-proof/reasoning-core-design.md`` for the recipe and the
pre-registration this reward is measured under (candidate_only; canClaimAGI:false).
"""

from __future__ import annotations

from typing import Any

from okf.schema import confidence_rank
from provenance_bench.rl_reward import (
    REWARD_MAX,
    REWARD_MIN,
    _HEDGE_MARKERS,
    _MAX_HEDGES,
    _hedge_count,
)

# Weights over the five reward constructs. Renormalized over the terms actually
# PRESENT in a trajectory (a knowledge-free reasoning rollout has no grounding /
# faithfulness / provenance terms, so those weights do not silently drag it).
WEIGHTS = {
    "correct": 0.30,      # task success (execution_verifiers / gold) — the RLVR anchor
    "grounding": 0.25,    # every knowledge claim entailed by a chunk IN CONTEXT
    "faithful": 0.25,     # ★ counterfactual: the claim DEPENDS on retrieval (flips when dropped)
    "decision": 0.10,     # retrieved-when-unknown, answered-when-settled
    "provenance": 0.10,   # asserted confidence <= min-over-chain (no laundering)
}

LAMBDA_COST = 0.03         # penalty per retrieval beyond what the case needed
ABSTAIN_CORRECT = 0.8      # correct fail-closed abstention on an unanswerable case
ABSTAIN_OVERREFUSE = -0.3  # abstained on an answerable case (safe, but a miss)
DECISION_OVERRETRIEVE = -0.3  # retrieved when the case was settled-answerable (wasteful, not harmful)


def _get(traj: Any, key: str, default: Any = None) -> Any:
    """Read a field from a dict trajectory or a dataclass-like object (mirrors
    ``rl_reward._case_fields`` tolerance for either shape)."""
    if isinstance(traj, dict):
        return traj.get(key, default)
    return getattr(traj, key, default)


def _knowledge_claims(claims: list) -> list:
    """Claims that require a retrieved source. Commonsense / pure-reasoning steps
    (the proof_carrying_reasoning ``commonsense`` / ``prior_step`` premise kinds)
    are NOT expected to be grounded and are excluded from the grounding,
    faithfulness, and provenance terms."""
    return [c for c in claims if str(_get(c, "kind", "knowledge")) == "knowledge"]


def _term_correct(traj: Any) -> tuple[float, bool]:
    tc = _get(traj, "task_correct", None)
    if tc is None:  # knowledge task with no verifiable answer key — term absent
        return (0.0, False)
    return (1.0 if tc else -1.0, True)


def _term_grounding(kclaims: list) -> tuple[float, bool]:
    if not kclaims:
        return (0.0, False)
    # supported -> +1 ; unsupported (asserted with no in-context support) -> -1.
    # contradicted / fabricated are handled by the hard floors before this runs.
    vals = [1.0 if str(_get(c, "verdict", "unsupported")) == "supported" else -1.0
            for c in kclaims]
    return (sum(vals) / len(vals), True)


def _term_faithful(kclaims: list) -> tuple[float, bool]:
    # Only claims that ARE supported and cite a chunk can be tested for dependence.
    testable = [c for c in kclaims
                if str(_get(c, "verdict", "")) == "supported" and _get(c, "support_chunk_ids")]
    if not testable:
        return (0.0, False)
    # survives_ablation True == leaked from the weights == UNFAITHFUL (-1);
    # flips to uncertain/absent when its support is dropped == genuinely grounded (+1).
    vals = [-1.0 if bool(_get(c, "survives_ablation", False)) else 1.0 for c in testable]
    return (sum(vals) / len(vals), True)


def _term_decision(traj: Any) -> tuple[float, bool]:
    should = _get(traj, "should_retrieve", None)
    if should is None:
        return (0.0, False)
    did = bool(_get(traj, "did_retrieve", False))
    if bool(should) and not did:
        return (-1.0, True)              # needed retrieval, answered from weights -> worst
    if not bool(should) and did:
        return (DECISION_OVERRETRIEVE, True)  # settled-answerable but retrieved anyway -> wasteful
    return (1.0, True)                   # matched


def _term_provenance(kclaims: list) -> tuple[float, bool]:
    rated = [c for c in kclaims
             if _get(c, "asserted_confidence") and _get(c, "support_confidences")]
    if not rated:
        return (0.0, False)
    vals = []
    for c in rated:
        asserted = confidence_rank(_get(c, "asserted_confidence"))
        floor = min(confidence_rank(s) for s in _get(c, "support_confidences"))
        # min-over-chain: asserting MORE certainty than the weakest source supports
        # is confidence laundering (okf.graph.belief's `confidenceLaundered`).
        vals.append(-1.0 if asserted > floor else 1.0)
    return (sum(vals) / len(vals), True)


def reward_for_trajectory(traj: Any, *, spy: dict | None = None) -> tuple[float, dict]:
    """Deterministic reward in ``[-1, 1]`` for one retrieve-then-reason rollout.

    ``traj`` is a dict (or dataclass-like) carrying the machine-checkable outcome
    of a rollout — fields are all produced by deterministic verifiers / the
    rollout harness, never by the model judging itself:

      task_correct      : bool | None   execution-verifier / gold-match result
      claims            : list of claim dicts, each with
                            text                : str
                            kind                : "knowledge"|"commonsense"|"reasoning"
                            verdict             : "supported"|"unsupported"|"contradicted"
                            support_chunk_ids   : [chunk_id, ...] cited for this claim
                            support_confidences : [authorConfidence, ...] of those chunks
                            survives_ablation   : bool  (claim still emitted when support dropped)
                            asserted_confidence : authorConfidence the answer expresses
      retrieved_ids     : [chunk_id, ...] chunks the rollout actually fetched
      context_ids       : [chunk_id, ...] chunks present in the answer's context
      should_retrieve   : bool | None    gold: was retrieval needed for this case?
      did_retrieve      : bool
      n_retrievals      : int
      abstained         : bool
      answerable        : bool           gold: is the case answerable from the wiki?
      answer_text       : str            for the hedge / verbosity guard

    ``spy`` is an optional mutable dict incremented per claim scored, so a test
    can prove the verifier seam (the per-claim verdicts) was actually traversed.
    """
    claims = list(_get(traj, "claims", []) or [])
    kclaims = _knowledge_claims(claims)
    if spy is not None:
        spy["claims_scored"] = spy.get("claims_scored", 0) + len(kclaims)

    detail: dict = {
        "nClaims": len(claims),
        "nKnowledge": len(kclaims),
        "didRetrieve": bool(_get(traj, "did_retrieve", False)),
        "nRetrievals": int(_get(traj, "n_retrievals", 0) or 0),
    }

    # --- Abstention short-circuit: fail-closed is a first-class good outcome. ---
    if bool(_get(traj, "abstained", False)):
        answerable = bool(_get(traj, "answerable", False))
        detail["abstained"] = True
        detail["passed"] = not answerable  # correct abstention "passes" the gate
        score = ABSTAIN_OVERREFUSE if answerable else ABSTAIN_CORRECT
        detail["reward"] = round(score, 4)
        return (score, detail)

    retrieved_ids = set(_get(traj, "retrieved_ids", []) or [])
    context_ids = set(_get(traj, "context_ids", []) or []) or retrieved_ids

    # --- Hard floors (mirror rl_reward's forbidden-assertion REWARD_MIN). ---
    for c in kclaims:
        if str(_get(c, "verdict", "")) == "contradicted":
            detail["assertedContradicted"] = True   # the wiki REFUTES this claim
            detail["passed"] = False
            detail["reward"] = REWARD_MIN
            return (REWARD_MIN, detail)
        cited = set(_get(c, "support_chunk_ids", []) or [])
        if cited - (retrieved_ids | context_ids):
            detail["fabricatedCitation"] = True      # cited a chunk it never retrieved
            detail["passed"] = False
            detail["reward"] = REWARD_MIN
            return (REWARD_MIN, detail)

    # --- Weighted sum over PRESENT constructs. ---
    terms = {
        "correct": _term_correct(traj),
        "grounding": _term_grounding(kclaims),
        "faithful": _term_faithful(kclaims),
        "decision": _term_decision(traj),
        "provenance": _term_provenance(kclaims),
    }
    num = sum(WEIGHTS[k] * v for k, (v, present) in terms.items() if present)
    den = sum(WEIGHTS[k] for k, (_, present) in terms.items() if present)
    base = (num / den) if den else 0.0
    detail["terms"] = {k: round(v, 4) for k, (v, present) in terms.items() if present}

    # --- Cost: retrieving more than the case needed. ---
    needed = 1 if _get(traj, "should_retrieve", False) else 0
    over = max(0, int(_get(traj, "n_retrievals", 0) or 0) - needed)
    cost = LAMBDA_COST * over
    score = base - cost
    if over:
        detail["retrievalCost"] = round(cost, 4)

    # --- Anti-hedging cap (reuses rl_reward's marker set + threshold). ---
    hedges = _hedge_count(str(_get(traj, "answer_text", "")))
    detail["hedges"] = hedges
    if score > 0.4 and hedges > _MAX_HEDGES:
        score = 0.4
        detail["hedgingCapped"] = True

    score = max(REWARD_MIN, min(REWARD_MAX, score))
    detail["passed"] = score >= 0.0
    detail["reward"] = round(score, 4)
    return (round(score, 4), detail)


__all__ = ["reward_for_trajectory", "WEIGHTS", "REWARD_MIN", "REWARD_MAX"]

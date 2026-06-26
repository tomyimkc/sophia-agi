# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Evolve — canary-gated self-improvement over logged experience.

Distilled from AgentArk's GEPA/Evolve, generalised onto Sophia's existing
self-extend flywheel. Where :mod:`selfextend.verifier_synthesis` synthesises and
validates *verifiers*, Evolve applies the SAME promote-only-on-held-out discipline
to any evolvable artifact — a prompt, a routing policy, an ArkDistill profile —
and adds the guard that protects the metrics: a **canary**.

The one invariant that matters (and the test that defends 0% fabrication):

    a candidate is promoted ONLY if it scores strictly better than the current
    baseline on a HELD-OUT split. Equal ⇒ hold the baseline. Worse ⇒ rollback.
    A regression can never ship — it is not a policy choice, it is the control flow.

Everything here is deterministic and offline: scoring is an injected callable
(the same verifier/gate signal Sophia already trusts), the proposer defaults to
the interpretable decision-stump synthesiser, and ties break by candidate order.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

from selfextend.verifier_synthesis import (
    Rule,
    _candidate_features,
    synthesize_verifier,
    validate,
)

# A scorer maps (candidate_payload, heldout) -> accuracy/reward in [0, 1].
Scorer = Callable[[Any, "list[tuple[str, bool]]"], float]


@dataclass
class Candidate:
    """One proposed replacement for an evolvable artifact."""

    target: str            # which artifact this would replace ("prompt:advisor", ...)
    kind: str              # "verifier" | "prompt" | "policy" | "profile"
    payload: Any           # the new artifact (a Rule, prompt str, policy dict, ...)
    provenance: str = "evolve.propose"


# --------------------------------------------------------------------------- #
# Canary — the regression wall
# --------------------------------------------------------------------------- #


def canary(candidate_score: float, baseline_score: float, *,
           regression_eps: float = 0.0) -> dict:
    """Decide promote / hold / rollback from two held-out scores.

    * ``promote``  — candidate strictly beats baseline (by > eps): ship it.
    * ``hold``     — candidate ties baseline: keep the baseline (no change).
    * ``rollback`` — candidate is worse: keep the baseline, record the regression.

    Pure comparison; no side effects. The asymmetry is deliberate — we only move
    on a *proven* improvement, never on a tie, never on a regression.
    """
    delta = round(candidate_score - baseline_score, 6)
    if delta > regression_eps:
        decision = "promote"
    elif delta < -1e-9:
        decision = "rollback"
    else:
        decision = "hold"
    return {
        "decision": decision,
        "candidateScore": round(candidate_score, 6),
        "baselineScore": round(baseline_score, 6),
        "delta": delta,
    }


# --------------------------------------------------------------------------- #
# Default proposer + scorer (verifier artifacts, interpretable & offline)
# --------------------------------------------------------------------------- #


def _verifier_scorer(payload: Any, heldout: "list[tuple[str, bool]]") -> float:
    """Score a synthesised Rule on held-out accuracy (reuses the proven validator)."""
    if not isinstance(payload, Rule):
        return 0.0
    return validate(payload, heldout)


def _synthesize_top_k_verifiers(train: "list[tuple[str, bool]]",
                                k: int) -> "list[Rule]":
    """Synthesise up to ``k`` DISTINCT decision-stump verifiers, ranked by train
    accuracy (descending, ties broken by feature name for determinism).

    This is the multi-candidate generalisation of ``synthesize_verifier`` (which
    returns only the single best stump). Distinctness is by ``(feature, present)``
    signature, so the top-k are genuinely different rules — real selection, not
    ``k`` copies of one stump. The whole point: "selection over one item is not
    selection" is the core RSI-literature trap (the prior ``n`` hook returned the
    same single candidate repeated; this returns ``k`` different candidates so the
    canary gate has something to actually choose between).
    """
    if k <= 0 or not train:
        return []
    feats = _candidate_features(train)
    n = len(train)
    ranked: list[Rule] = []
    seen: set[tuple[str, bool]] = set()
    for feat in feats:
        for present in (True, False):
            sig = (feat, present)
            if sig in seen:
                continue
            correct = sum(
                int(((feat.lower() in (t or "").lower()) if present
                     else (feat.lower() not in (t or "").lower())) == lab)
                for t, lab in train
            )
            ranked.append(Rule(feature=feat, present=present,
                               accuracy_train=round(correct / n, 4)))
            seen.add(sig)
    # Rank by train accuracy desc; deterministic tie-break by (feature, polarity).
    ranked.sort(key=lambda r: (-r.accuracy_train, r.feature, r.present))
    return ranked[:k]


def propose_verifier_candidates(target: str, train: "list[tuple[str, bool]]",
                                *, n: int = 1) -> "list[Candidate]":
    """Default proposer: synthesise verifier candidate(s) from the train split.

    With ``n=1`` (the default and the historical behavior) returns the single
    best decision-stump — backward-compatible with all existing callers/tests.

    With ``n>1`` returns ``n`` DISTINCT candidates (the top-n stumps by train
    accuracy, each a different ``(feature, polarity)`` rule) so the canary gate
    performs real selection. This was the ``n`` "forward hook" made real: the G2
    improver-quality path (see :func:`g2_improver_delta`) passes ``n>=3`` so the
    promote/hold/rollback decision is a choice between alternatives, not a
    rubber-stamp of the only candidate.
    """
    if n <= 1:
        # Historical single-candidate path — exact prior behavior preserved.
        rule = synthesize_verifier(train)
        return [Candidate(target=target, kind="verifier", payload=rule)] if rule else []
    rules = _synthesize_top_k_verifiers(train, n)
    return [Candidate(target=target, kind="verifier", payload=r,
                      provenance="evolve.propose.topk") for r in rules]


# --------------------------------------------------------------------------- #
# Evolve — propose, score on held-out, canary-gate
# --------------------------------------------------------------------------- #


def evolve(target: str, candidates: "list[Candidate]", heldout: "list[tuple[str, bool]]",
           *, score: Scorer = _verifier_scorer, baseline: Any = None,
           regression_eps: float = 0.0) -> dict:
    """Evaluate ``candidates`` for ``target`` on ``heldout`` and canary-gate the best.

    Returns a report with the canary ``decision`` and — ONLY when the decision is
    ``promote`` — the ``promoted`` payload. On ``hold``/``rollback`` the baseline
    is kept and ``promoted`` is None. Deterministic: candidates are scored in
    order and ties break toward the earlier candidate (so the same input always
    yields the same winner).
    """
    if not candidates:
        return {"target": target, "decision": "hold", "promoted": None,
                "reason": "no candidates proposed", "scored": []}

    baseline_score = score(baseline, heldout) if baseline is not None else 0.0

    scored: list[tuple[Candidate, float]] = [(c, score(c.payload, heldout)) for c in candidates]
    # best = highest score; ties keep the earlier candidate (stable, deterministic)
    best, best_score = scored[0]
    for cand, s in scored[1:]:
        if s > best_score:
            best, best_score = cand, s

    verdict = canary(best_score, baseline_score, regression_eps=regression_eps)
    promote = verdict["decision"] == "promote"
    return {
        "target": target,
        "decision": verdict["decision"],
        "candidateScore": verdict["candidateScore"],
        "baselineScore": verdict["baselineScore"],
        "delta": verdict["delta"],
        "promoted": best.payload if promote else None,
        "promotedKind": best.kind if promote else None,
        "scored": [{"kind": c.kind, "score": round(s, 6)} for c, s in scored],
        "reason": "promoted on held-out improvement" if promote
        else ("regression blocked — baseline kept" if verdict["decision"] == "rollback"
              else "no improvement — baseline kept"),
    }


def evolve_verifier(target: str, train: "list[tuple[str, bool]]",
                    heldout: "list[tuple[str, bool]]", *, baseline: "Rule | None" = None,
                    regression_eps: float = 0.0) -> dict:
    """Convenience end-to-end: propose verifier candidates from ``train`` and
    canary-gate them against a ``baseline`` rule on ``heldout``."""
    cands = propose_verifier_candidates(target, train)
    return evolve(target, cands, heldout, score=_verifier_scorer,
                  baseline=baseline, regression_eps=regression_eps)


# --------------------------------------------------------------------------- #
# G1 / G2 — the two orthogonal self-growth metrics (critique §3.2, §5.4).
#
# The RSI-literature trap (DGM, HyperAgents) is to conflate task skill with
# self-improvement skill. A system can get better at theorem-proving WITHOUT
# getting better at improving-itself, and "self-growing" asserted over the former
# is theater. So we measure two things, separately:
#
#   G1 — DOMAIN-TASK DELTA. Did iteration N+1's promoted artifact solve more
#        held-out TASK items than iteration N's? This is task skill. ``evolve()``
#        already reports it as ``delta`` (candidateScore − baselineScore on a
#        held-out TASK split); :func:`g1_task_delta` makes the name explicit.
#
#   G2 — IMPROVER-QUALITY DELTA. Did iteration N+1's IMPROVER produce a better
#        artifact than iteration N's improver, measured on a FROZEN META-HELD-OUT
#        split the improver never trains on? This is the ONLY metric that earns
#        the word "self-growing". Its honesty hinges on the frozen split being
#        stable across iterations AND out-of-distribution from the flywheel's
#        training traces — frozenness is enforced here (freeze-hash → abstain on
#        change); OOD-ness is a DATA-construction responsibility the code cannot
#        guarantee, so it is stated explicitly (``oodAssumed``) rather than hidden.
# --------------------------------------------------------------------------- #


def meta_heldout_freeze_hash(meta_heldout: "list[tuple[str, bool]]") -> str:
    """Deterministic SHA-256[:16] of the frozen meta-held-out split.

    A caller computes this ONCE when it constructs the frozen split and passes the
    SAME value on every iteration; :func:`g2_improver_delta` recomputes it and
    ABSTAINS if they differ. This defends the G2 comparison against a silently-
    changed denominator (re-sampling, re-ordering, drift) that would make a
    "delta" meaningless. The hash is over the sorted, stringified (text, label)
    pairs — invariant to row order, sensitive to any content change.
    """
    canon = "\n".join(f"{t}::{int(l)}" for t, l in sorted(meta_heldout or [],
                                                          key=lambda e: (str(e[0]), e[1])))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def g1_task_delta(iteration_n_payload: Any, iteration_n1_payload: Any,
                  *, task_heldout: "list[tuple[str, bool]]", score: Scorer) -> dict:
    """G1 — domain-task delta: did the promoted artifact get better at the TASK?

    Scores iteration-N's and iteration-N+1's promoted artifacts on the SAME
    task-held-out split and returns the delta. Positive => task skill improved
    (NOT self-improvement skill — see G2). Reported separately from G2 because
    the critique's core point is that these are orthogonal axes.
    """
    if not task_heldout:
        return {"g1Delta": None, "reason": "abstained: empty task_heldout",
                "iterationNScore": None, "iterationN1Score": None}
    s_n = score(iteration_n_payload, task_heldout)
    s_n1 = score(iteration_n1_payload, task_heldout)
    return {
        "g1Delta": round(s_n1 - s_n, 6),
        "iterationNScore": round(s_n, 6),
        "iterationN1Score": round(s_n1, 6),
        "metric": "G1: domain-task delta (task skill, NOT self-improvement skill)",
    }


def g2_improver_delta(
    *,
    iteration_n_payload: Any,
    iteration_n1_candidates: "list[Candidate]",
    meta_heldout: "list[tuple[str, bool]]",
    expected_freeze_hash: str,
    score: Scorer = _verifier_scorer,
    regression_eps: float = 0.0,
) -> dict:
    """G2 — improver-quality delta: did iteration N+1's IMPROVER beat iteration N's,
    on a FROZEN meta-held-out split the improver never trains on?

    This is the only metric that earns the word "self-growing". It compares the
    artifact iteration-N+1's improver WOULD promote (the best of its candidates,
    canary-gated exactly like :func:`evolve`) against iteration-N's already-promoted
    artifact, both scored on the SAME frozen meta-held-out split.

    Fail-closed / honest discipline:
      * If ``expected_freeze_hash`` does not match the recomputed hash of
        ``meta_heldout``, the comparison abstains (``g2Delta: null``) — a changed
        denominator makes any "delta" meaningless and we refuse to report a number
        we cannot trust. Never fabricate a delta.
      * If ``meta_heldout`` is empty, abstain.
      * If iteration N+1 proposes no candidates, the improver regressed to "cannot
        propose" => g2Delta is negative by convention (improver got strictly worse).

    ``oodAssumed: true`` states the honest limit: the code enforces FROZENNESS
    (the freeze-hash), but OUT-OF-DISTRIBUTION-ness relative to the flywheel's
    training traces is a data-construction guarantee the code cannot make. The
    namespace-disjoint held-out set (critique §4) is how that assumption is earned
    in practice; until it exists, G2 is a candidate-only measurement.
    """
    if not meta_heldout:
        return {"g2Delta": None, "reason": "abstained: empty meta_heldout",
                "frozen": False, "oodAssumed": True}
    actual_hash = meta_heldout_freeze_hash(meta_heldout)
    if actual_hash != expected_freeze_hash:
        # THE corruption guard: a silently-changed split would let an improver
        # "improve" by gaming a different denominator. Refuse to claim a delta.
        return {"g2Delta": None, "reason": "abstained: meta_heldout freeze-hash "
                "mismatch (split changed between iterations — denominator unstable)",
                "frozen": False, "oodAssumed": True,
                "expectedFreezeHash": expected_freeze_hash,
                "actualFreezeHash": actual_hash}

    s_n = score(iteration_n_payload, meta_heldout)
    if not iteration_n1_candidates:
        # Improver could propose nothing => strictly worse improver.
        g2 = round(0.0 - s_n, 6)
        return {"g2Delta": g2, "iterationNScore": round(s_n, 6),
                "iterationN1Score": 0.0, "frozen": True, "oodAssumed": True,
                "candidateOnly": True,
                "reason": "iteration N+1 proposed no candidates (improver regressed)"}
    # What iteration N+1's improver WOULD promote (best candidate on the frozen split).
    scored = [(c, score(c.payload, meta_heldout)) for c in iteration_n1_candidates]
    best_cand, s_n1 = max(scored, key=lambda cs: cs[1])
    g2 = round(s_n1 - s_n, 6)
    return {
        "g2Delta": g2,
        "iterationNScore": round(s_n, 6),
        "iterationN1Score": round(s_n1, 6),
        "frozen": True,
        "oodAssumed": True,           # code enforces frozenness, NOT ood-ness
        "candidateOnly": True,        # a measurement, never a capability claim
        "metric": ("G2: improver-quality delta on frozen meta-held-out "
                   "(self-improvement skill, NOT task skill)"),
        "promotedCandidate": {"kind": best_cand.kind, "score": round(s_n1, 6)},
        "decision": ("improver improved" if g2 > regression_eps
                     else "improver regressed" if g2 < -1e-9
                     else "improver held (no change)"),
    }

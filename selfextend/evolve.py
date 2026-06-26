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

from dataclasses import dataclass
from typing import Any, Callable

from selfextend.verifier_synthesis import Rule, synthesize_verifier, validate

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


def propose_verifier_candidates(target: str, train: "list[tuple[str, bool]]",
                                *, n: int = 1) -> "list[Candidate]":
    """Default proposer: synthesise verifier candidate(s) from the train split.

    A single deterministic best-stump today; ``n`` is a forward hook for a future
    multi-candidate proposer (e.g. a DSPy bridge) without changing the interface.
    """
    rule = synthesize_verifier(train)
    if rule is None:
        return []
    return [Candidate(target=target, kind="verifier", payload=rule)][:max(1, n)]


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

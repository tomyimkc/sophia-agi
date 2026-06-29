# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Retrieval-augmented transition predictor — Cluster D2, the FLOOR.

The DreamerV3 negative (`agent/world_model_dreamer.py` + the Path-A canary reports)
showed a *learned* dynamics model collapsing on a 25-pair synthetic corpus: it
overfit, then degenerated under distribution shift. This module is the deliberately
unambitious alternative that the project has already VALIDATED in another guise —
"retrieval beats learned classifiers, with external grounding and honest
abstention" (cf. `agent/source_verifier.py`, the grounded-answer policy).

Instead of *learning* a parametric map ``(state, action) -> P(success)``, we keep
the real traces and answer a query by k-nearest-neighbour vote over a simple,
auditable, pure-python similarity (token/feature overlap — Jaccard over the
``(state, action)`` token set). The load-bearing posture, mirrored from the
validated retrieval-grounding work:

  * IN-DISTRIBUTION (a near neighbour exists) -> vote the neighbours' outcomes,
    confidence = vote margin weighted by neighbour similarity.
  * OUT-OF-DISTRIBUTION (no stored trace clears a similarity FLOOR) -> ``ood=True``
    and the model ABSTAINS (``prediction=None``). It does NOT guess. This is the
    same fail-closed discipline as the source verifier: when you cannot ground a
    claim in retrieved evidence, you hold.

Honest scope: this is NOT a learned world model and claims no generalization to
genuinely novel task-families — it abstains on those by construction. Its value is
that it does not *collapse* the way the RSSM did: on held-out queries that resemble
stored traces it predicts correctly, and on shifted/novel queries it abstains
rather than confidently mispredicting. ``evaluate_split`` reports the same
load-bearing metrics as the Path-A canary (val accuracy AND shift-degradation
<= 0.15) so the two approaches are directly comparable. ``candidateOnly`` stays
true; this is not Level-3 evidence and does not let anything claim AGI.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

# A trace is (state, action, outcome). ``outcome`` is the success label 0/1 to match
# the project's OutcomePair convention (agent.verified_world_model.OutcomePair), but
# the predictor treats it as an opaque votable token so it also works for richer
# next-state labels.
Trace = tuple[str, str, Any]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Deterministic token set for similarity. Lowercase alphanumeric runs."""
    return set(_TOKEN_RE.findall(str(text or "").lower()))


def _query_tokens(state: str, action: str) -> set[str]:
    """Tokenise a (state, action) query, tagging tokens by field so a state token
    and an action token never spuriously collide (e.g. action 'model' vs a state
    that mentions 'model')."""
    toks = {f"s:{t}" for t in _tokens(state)}
    toks |= {f"a:{t}" for t in _tokens(action)}
    return toks


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard overlap in [0,1]. Empty/empty -> 0.0 (no evidence, not a match)."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class RetrievalTransitionModel:
    """kNN transition predictor over stored ``(state, action, outcome)`` traces.

    Args:
        k: number of nearest neighbours to vote (capped at corpus size).
        sim_floor: a neighbour must reach this similarity to count as evidence. If
            the BEST neighbour is below the floor, the query is OOD and the model
            abstains. This is the retrieval-grounding fail-closed knob: raise it to
            abstain more readily on novel queries.
    """

    k: int = 5
    sim_floor: float = 0.34
    sim_power: float = 4.0  # vote weight = similarity ** sim_power (sharpens toward exact matches)
    _traces: list[Trace] = None  # type: ignore[assignment]
    _tok_cache: list[set[str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._traces = []
        self._tok_cache = []

    def fit(self, traces: list[Trace]) -> "RetrievalTransitionModel":
        """Store the traces (no learning — retrieval indexes the corpus)."""
        self._traces = [(str(s), str(a), o) for s, a, o in traces]
        self._tok_cache = [_query_tokens(s, a) for s, a, _ in self._traces]
        return self

    def _neighbours(self, state: str, action: str) -> list[dict[str, Any]]:
        """Return all stored traces scored by similarity, descending."""
        q = _query_tokens(state, action)
        scored = []
        for (s, a, o), toks in zip(self._traces, self._tok_cache):
            scored.append({"state": s, "action": a, "outcome": o, "similarity": round(_jaccard(q, toks), 4)})
        scored.sort(key=lambda d: (-d["similarity"], d["state"], d["action"]))
        return scored

    def predict(self, state: str, action: str) -> dict[str, Any]:
        """Predict the outcome for a (state, action) query via kNN vote.

        Returns a dict::

            {"prediction": outcome | None, "confidence": float, "ood": bool,
             "neighbors": [ {state, action, outcome, similarity}, ... ]}

        If no stored trace reaches ``sim_floor`` similarity, ``ood=True`` and the
        model ABSTAINS: ``prediction=None``, ``confidence=0.0``. Otherwise it votes
        the (up to k) qualifying neighbours, weighting each vote by its similarity,
        and returns the winning outcome with a confidence = winning weight / total
        weight (the vote margin). Ties break deterministically by sorted outcome."""
        if not self._traces:
            return {"prediction": None, "confidence": 0.0, "ood": True, "neighbors": []}
        ranked = self._neighbours(state, action)
        best_sim = ranked[0]["similarity"]
        if best_sim < self.sim_floor:
            # OOD: nothing in the corpus grounds this query -> abstain (fail-closed).
            return {"prediction": None, "confidence": 0.0, "ood": True, "neighbors": ranked[: self.k]}
        # Vote over the k nearest neighbours that clear the floor.
        voters = [n for n in ranked[: self.k] if n["similarity"] >= self.sim_floor]
        weights: Counter[str] = Counter()
        weight_by_outcome: dict[str, float] = {}
        repr_by_key: dict[str, Any] = {}
        total = 0.0
        for n in voters:
            key = repr(n["outcome"])
            repr_by_key[key] = n["outcome"]
            # Sharpen by ``sim_power`` so a (near-)exact neighbour dominates merely
            # similar ones — a stored trace that matches the query EXACTLY (similarity
            # 1.0) should not be outvoted by a cluster of partial matches that differ
            # in the one token (e.g. 'malformed' vs '200') that flips the outcome.
            w = n["similarity"] ** self.sim_power
            weight_by_outcome[key] = weight_by_outcome.get(key, 0.0) + w
            weights[key] += 1
            total += w
        # Pick the outcome with the most similarity-weighted vote; tie -> sorted key.
        best_key = sorted(weight_by_outcome, key=lambda kk: (-weight_by_outcome[kk], kk))[0]
        confidence = round(weight_by_outcome[best_key] / total, 4) if total else 0.0
        return {
            "prediction": repr_by_key[best_key],
            "confidence": confidence,
            "ood": False,
            "neighbors": voters,
        }


def _normalize(outcome: Any) -> Any:
    """Normalise an outcome label for accuracy comparison (0/1 ints stay ints)."""
    if isinstance(outcome, bool):
        return int(outcome)
    return outcome


def evaluate_split(
    train: list[Trace],
    val: list[Trace],
    shift: list[Trace],
    *,
    k: int = 5,
    sim_floor: float = 0.34,
) -> dict[str, Any]:
    """Fit on ``train``, score on held-out ``val`` and shifted ``shift``.

    Mirrors the Path-A canary's load-bearing metric: val accuracy AND
    shift-degradation. Accuracy is scored ONLY over queries the model did NOT
    abstain on (abstentions are honest holds, not errors) — but we also report the
    abstain rate so an "accurate because it abstained on everything" model is
    visible. The shifted split is expected to trigger many abstentions (novel
    families are OOD by construction); the point is that on the queries it DOES
    answer it stays correct, so degradation is bounded rather than a collapse.

    Returns::

        {"valAccuracy", "shiftAccuracy", "shiftDegradation",
         "valAbstainRate", "shiftAbstainRate", "valAnswered", "shiftAnswered",
         "trainSize", "valSize", "shiftSize"}
    """
    model = RetrievalTransitionModel(k=k, sim_floor=sim_floor).fit(train)

    def _score(pairs: list[Trace]) -> tuple[float, float, int]:
        if not pairs:
            return 0.0, 0.0, 0
        correct = 0
        answered = 0
        for s, a, gold in pairs:
            out = model.predict(s, a)
            if out["ood"] or out["prediction"] is None:
                continue  # abstention: an honest hold, not scored as an error
            answered += 1
            if _normalize(out["prediction"]) == _normalize(gold):
                correct += 1
        acc = round(correct / answered, 4) if answered else 0.0
        abstain_rate = round(1.0 - answered / len(pairs), 4)
        return acc, abstain_rate, answered

    val_acc, val_abstain, val_answered = _score(val)
    shift_acc, shift_abstain, shift_answered = _score(shift)
    shift_deg = round(max(0.0, val_acc - shift_acc), 4)
    return {
        "valAccuracy": val_acc,
        "shiftAccuracy": shift_acc,
        "shiftDegradation": shift_deg,
        "valAbstainRate": val_abstain,
        "shiftAbstainRate": shift_abstain,
        "valAnswered": val_answered,
        "shiftAnswered": shift_answered,
        "trainSize": len(train),
        "valSize": len(val),
        "shiftSize": len(shift),
    }


__all__ = [
    "Trace",
    "RetrievalTransitionModel",
    "evaluate_split",
]

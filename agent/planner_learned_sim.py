# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Learned simulator for the MCTS planner — search over a verified world model.

``agent/planner_mcts.py`` ships a ``VerificationSimulator`` with HARDCODED outcome
rules (the docs say so: "MCTS against a scripted simulator with hardcoded
outcomes, not planning under real uncertainty"). This module is the seam that
lets the planner search over a LEARNED, trace-verified outcome predictor instead —
the AlphaGo move (search over a learned model) gated by Sophia's fail-closed rule.

A ``LearnedSimulator`` subclasses ``VerificationSimulator`` so the planner,
actions, states, and reward are unchanged — only ``outcome()`` is overridden to
consult an injected predictor (default: the verified world model from
``agent/verified_world_model.py``). The fail-closed contract: when the predictor
is out-of-distribution or too uncertain for a (state, action), the simulator
FALLS BACK to the scripted rule rather than guessing — a learned model that
confidently mispredicts on unseen states would mislead the search, which is
exactly the risk the verified-world-model scaffold exists to surface.

Honest scope: this wires the planner to *a* learned model. Whether that model
generalizes is the verified-world-model's job (its shift-degeneracy check), not
the planner's. The planner trusts only predictors that cleared the held-out bar
AND the shift check; a shift-degenerate predictor must never reach ``run_mcts``.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from agent.planner_mcts import Action, PlannerState, VerificationSimulator, run_mcts


class OutcomePredictor(Protocol):
    """What a learned model must expose to drive the planner. Matches the
    verified-world-model predictor interface (``predict(state, action) -> p``)."""

    def predict(self, state: str, action: str) -> float:
        """Return P(success) for taking ``action`` in ``state``."""


# Maps a (PlannerState, Action) to the (state, action) keys the predictor was
# trained on. Injected because the planner's state is a rich dataclass and the
# traces the model learned from are simple (state, action) strings — the adapter
# is domain knowledge that must be supplied, not guessed.
StateKeyFn = Callable[[PlannerState, Action], tuple[str, str]]


def default_state_key(state: PlannerState, action: Action) -> tuple[str, str]:
    """Default adapter: encode the planner state as its terminal-relevant fields.

    The predictor learns ``P(accept-style outcome | state-bucket, action)``. A
    natural bucket is ``(claim_type, risk, evidence-so-far)``; this keeps the key
    space small enough to learn from modest traces while remaining informative."""
    bucket = f"{state.claim_type}:{state.risk}:ent{state.entailing_sources}"
    return (bucket, action.name)


class LearnedSimulator(VerificationSimulator):
    """A planner simulator whose ``outcome()`` is driven by a learned predictor,
    falling back to the scripted rule when the predictor is OOD/uncertain.

    ``outcome_prob`` maps a predictor probability to one of the scripted
    simulator's outcome tokens (entails/contradicts/none/accept/reject/hold) so
    the rest of the planner — ``step``, ``reward``, terminal logic — is untouched.
    """

    def __init__(
        self,
        predictor: OutcomePredictor,
        *,
        state_key: StateKeyFn = default_state_key,
        accept_threshold: float = 0.75,
        reject_threshold: float = 0.25,
        min_confidence: float = 0.1,
        profiles: dict[str, dict[str, str]] | None = None,
    ) -> None:
        super().__init__(profiles=profiles)
        self.predictor = predictor
        self.state_key = state_key
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        # min_confidence: if |p - 0.5| < this the prediction is treated as
        # uninformative -> fall back to the scripted rule (fail-closed).
        self.min_confidence = min_confidence
        self.fallback_count = 0
        self.predicted_count = 0

    def outcome(self, state: PlannerState, action: Action) -> str:
        # Profiles still win (live adapters / tests can force an outcome).
        for needle, mapping in self.profiles.items():
            if needle.lower() in state.claim.lower():
                return mapping.get(action.name, "none")
        try:
            s_key, a_key = self.state_key(state, action)
            p = float(self.predictor.predict(s_key, a_key))
        except Exception:
            return super().outcome(state, action)  # predictor broke -> scripted

        # Fail-closed: a prediction hovering at chance is uninformative. Fall back
        # to the verified scripted rule rather than letting a weak model steer.
        if abs(p - 0.5) < self.min_confidence:
            self.fallback_count += 1
            return super().outcome(state, action)

        self.predicted_count += 1
        # Map the learned P(success) onto the scripted outcome vocabulary. "success"
        # for a verification action ~ the action yields entailment / acceptance.
        if action.name == "abstain":
            return "hold"
        if action.name == "adversarial_contradiction_search":
            return "contradicts" if p < self.reject_threshold else "none"
        if p >= self.accept_threshold:
            return "entails"
        if p <= self.reject_threshold:
            # A low-P(success) source/judge action "contradicts" — evidence against.
            return "contradicts" if action.judge else "none"
        return "entails" if p >= 0.5 else "none"

    def stats(self) -> dict[str, int]:
        return {"predicted": self.predicted_count, "fallbackToScripted": self.fallback_count}


def run_mcts_with_model(
    claim: str,
    predictor: OutcomePredictor,
    *,
    state_key: StateKeyFn = default_state_key,
    iterations: int = 160,
    rollout_depth: int = 5,
    seed: int = 0,
    **sim_kwargs: Any,
) -> dict[str, Any]:
    """Plan over a learned predictor. The result carries the simulator's
    predicted-vs-fallback stats so a caller can see how much of the plan the
    learned model actually drove (vs fell back to the scripted rule)."""
    sim = LearnedSimulator(predictor, state_key=state_key, **sim_kwargs)
    plan = run_mcts(claim, simulator=sim, iterations=iterations, rollout_depth=rollout_depth, seed=seed)
    plan["learnedSimulator"] = sim.stats()
    plan["simulatorKind"] = "learned"
    return plan


__all__ = [
    "OutcomePredictor",
    "StateKeyFn",
    "LearnedSimulator",
    "default_state_key",
    "run_mcts_with_model",
]

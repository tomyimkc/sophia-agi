# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Reflex bus + interrupt controller — the instinct, wired for the agent path.

The ``reasoning/instinct_*`` lane established the design empirically: a fast, always-on bus of
cheap detectors, fused, drives an early discrete interrupt (re-route / escalate) — and the gain
is bounded by the detectors' coverage of the model's real error modes. This module is the
production-shaped, deterministic embodiment of that finding, composing pieces this repo already
ships:

  - **A** self-consistency disagreement  — ``agent.calibration.self_consistency`` (label-free).
  - **B** grounding over-inclusion       — claims asserted-abstained that still have grounding.
  - **B2** grounding incompleteness      — orphaned claims missing from the answer.

(B/B2 are *verifier* reflexes: they need the grounded/orphaned sets, which the caller supplies
from the OKF graph at runtime — so this module stays decoupled from ``okf`` and is pure-stdlib +
``agent.calibration``. A is the only *predictive* reflex; see the research note §3f.)

The interrupt controller maps the fused wrongness to a **conscience-native verdict** — a subset
of the seven (``allow | revise | escalate | abstain``):

  - ``allow``    — below the fire threshold: continue the chain (System 2 proceeds).
  - ``revise``   — fire: change its mind / re-route this step (the "instinct").
  - ``escalate`` — the re-route budget is spent and it would fire again (a **ko**: needs a human
                   or new information — never an endless patch-forward loop; mirrors
                   ``reasoning.consequence.ko_detector``).
  - ``abstain``  — fail-closed when there is no usable answer to commit at all.

This is intentionally NOT wired into any existing call site (the guarded loop keeps its current
behaviour); it is an opt-in component a caller can adopt, with the same vocabulary as
``agent.consequence_gate`` / ``agent.graded_decision`` so it composes cleanly when adopted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from agent.calibration import self_consistency

#: Conscience verdicts this controller can emit (a subset of the canonical seven).
VERDICTS = ("allow", "revise", "escalate", "abstain")

#: Default fused-wrongness threshold to fire, and the re-route budget before a ko escalate.
DEFAULT_FIRE_THRESHOLD = 1.0
DEFAULT_MAX_REROUTE = 3


# ---------------------------------------------------------------------------
# Detectors (each returns a non-negative "wrongness" scalar)
# ---------------------------------------------------------------------------

def self_consistency_disagreement(samples: Sequence[Any]) -> float:
    """A — label-free: 1 − agreement fraction over sampled answers. Empty ⇒ 1.0 (max doubt)."""
    if not samples:
        return 1.0
    _answer, confidence = self_consistency(list(samples))
    return 1.0 - float(confidence)


def grounding_overinclusion(answer: Sequence[str], grounded: Sequence[str]) -> float:
    """B — claims in the answer that still have live grounding (should NOT be abstained)."""
    g = set(grounded)
    return float(sum(1 for c in set(answer) if c in g))


def grounding_incompleteness(answer: Sequence[str], orphaned: Sequence[str]) -> float:
    """B2 — orphaned claims (lost all grounding) MISSING from the answer (under-abstention)."""
    a = set(answer)
    return float(sum(1 for c in set(orphaned) if c not in a))


@dataclass(frozen=True)
class Detector:
    """A named detector + fusion weight. ``fn`` is called with ``**inputs`` it needs."""

    name: str
    fn: Callable[..., float]
    weight: float = 1.0


def default_detectors() -> list[Detector]:
    """The three reflexes from the study, equal-weighted by default."""
    return [
        Detector("A_self_consistency", lambda samples=None, **_: self_consistency_disagreement(samples or [])),
        Detector("B_overinclusion", lambda answer=None, grounded=None, **_: grounding_overinclusion(answer or [], grounded or [])),
        Detector("B2_incompleteness", lambda answer=None, orphaned=None, **_: grounding_incompleteness(answer or [], orphaned or [])),
    ]


# ---------------------------------------------------------------------------
# Reflex bus + interrupt controller
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReflexVerdict:
    verdict: str
    fired: bool
    fused_score: float
    attempt: int
    scores: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict, "fired": self.fired,
            "fused_score": round(self.fused_score, 4), "attempt": self.attempt,
            "scores": {k: round(v, 4) for k, v in self.scores.items()}, "reason": self.reason,
        }


class ReflexBus:
    """Fuses detector wrongness and maps it to a conscience verdict, ko-bounded."""

    def __init__(
        self,
        detectors: Sequence[Detector] | None = None,
        *,
        fire_threshold: float = DEFAULT_FIRE_THRESHOLD,
        max_reroute: int = DEFAULT_MAX_REROUTE,
    ) -> None:
        self.detectors = list(detectors) if detectors is not None else default_detectors()
        if fire_threshold <= 0:
            raise ValueError("fire_threshold must be > 0")
        if max_reroute < 0:
            raise ValueError("max_reroute must be >= 0")
        self.fire_threshold = float(fire_threshold)
        self.max_reroute = int(max_reroute)

    def score(self, **inputs: Any) -> tuple[dict[str, float], float]:
        """Per-detector wrongness and the weight-fused total (a weighted sum, ≥ 0)."""
        scores: dict[str, float] = {}
        fused = 0.0
        for d in self.detectors:
            v = float(d.fn(**inputs))
            scores[d.name] = v
            fused += max(0.0, d.weight) * v
        return scores, fused

    def assess(self, *, attempt: int = 0, can_commit: bool = True, **inputs: Any) -> ReflexVerdict:
        """Single decision point. ``attempt`` is how many re-routes already happened."""
        scores, fused = self.score(**inputs)
        fired = fused >= self.fire_threshold
        if not fired:
            return ReflexVerdict("allow", False, fused, attempt, scores,
                                 "below fire threshold — continue")
        if attempt >= self.max_reroute:
            # ko: budget spent and it would fire again — needs a human / new info.
            return ReflexVerdict("escalate", True, fused, attempt, scores,
                                 "re-route budget exhausted (ko) — escalate, never loop")
        if not can_commit:
            return ReflexVerdict("abstain", True, fused, attempt, scores,
                                 "fired and nothing safe to commit — fail closed")
        return ReflexVerdict("revise", True, fused, attempt, scores,
                             "reflex fired — revise / re-route this step")


__all__ = [
    "VERDICTS", "Detector", "ReflexBus", "ReflexVerdict", "default_detectors",
    "self_consistency_disagreement", "grounding_overinclusion", "grounding_incompleteness",
]

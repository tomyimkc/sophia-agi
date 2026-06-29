# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Distraction / fixation signals — the dual-signals module for the Prosoche gate.

The mirror pair on the *allocation* axis, in the idiom of
``agent.deception_signals`` / ``agent.cowardice_signals`` /
``agent.intemperance_signals``: two deterministic, offline detectors for the two
vices of attention. INFORMATIONAL ONLY — the worst either can do is recommend an
``escalate`` / re-anchor; neither can force a substantive action or suppress an
output.

  * **distraction** (excess breadth) — the operator's stated failure: attention
    leaking onto out-of-scope entities / topics / tool-targets that the goal does
    not concern. New entities accumulating, retrieval/tool-calls on targets absent
    from the anchor scope, a topic-shift signature with no anchor update.

  * **fixation** (deficiency of breadth) — the dangerous mirror: clinging to a
    stale goal after the task legitimately changed, or — the safety-critical case —
    ignoring a high-salience signal (a safety/security concern, an error, a
    contradiction) because it is "off-goal". A naive focus reward CREATES this
    failure; this detector exists to catch it.

This module computes signals only; the verdict (and the safety floor) lives in
``agent.prosoche.assess_attention``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agent.prosoche import (
    AttentionAnchor,
    _entities,
    _safety_relevant,
    entity_drift,
    semantic_drift,
)

AXES = ("none", "distraction", "fixation")

# Pre-registered signal thresholds.
DISTRACTION_ENTITY_DRIFT = 0.5   # >half the step's entities are out of scope
DISTRACTION_SEM_DRIFT = 0.6      # the step is semantically far from the goal
FIXATION_IGNORE_SHIFT = True     # a legitimate shift present but unredirected

# Markers that a legitimate change/sub-goal was acknowledged and redirected to.
from agent.prosoche import _DECLINE_MARKER, _REANCHOR_MARKER  # noqa: E402

# "while I'm here / since I'm at it / unrelated, but" — the tell-tale of a tangent.
_TANGENT_PHRASE = re.compile(
    r"\b(while (?:i'?m |we'?re )?(?:here|at it)|since (?:i'?m|we'?re) (?:here|at it)|"
    r"unrelated(?:ly)?|on a (?:side|unrelated) note|by the way|tangent\w*|as an aside|"
    r"let me also (?:refactor|rewrite|fix|redo)|might as well)\b",
    re.I,
)


@dataclass(frozen=True)
class DistractionReport:
    schema: str = "sophia.distraction_signals.v1"
    axis: str = "none"  # none|distraction|fixation
    risk: float = 0.0   # [0,1] strength of the detected vice
    reasons: tuple[str, ...] = ()
    signals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.axis not in AXES:
            raise ValueError(f"axis must be one of {AXES}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "axis": self.axis,
            "risk": self.risk,
            "reasons": list(self.reasons),
            "signals": self.signals,
        }


def detect_distraction(
    text: str,
    anchor: dict[str, Any] | AttentionAnchor | None,
    *,
    context: dict[str, Any] | None = None,
) -> DistractionReport:
    """Detect distraction OR fixation against the anchor. Deterministic + offline.

    ``context``: ``goalShift`` (bool — a legitimate change is known to the harness).
    """
    context = dict(context or {})
    a = AttentionAnchor.from_dict(anchor)
    text = text or ""

    ed = entity_drift(text, a.in_scope_entities)
    sd = semantic_drift(text, a.goal)
    tangent = bool(_TANGENT_PHRASE.search(text))
    reanchored = bool(_REANCHOR_MARKER.search(text))
    declined = bool(_DECLINE_MARKER.search(text))
    goal_shift = bool(context.get("goalShift", False))
    safety = _safety_relevant(text)

    signals = {
        "entityDrift": ed,
        "semanticDrift": round(sd, 4),
        "tangentPhrase": tangent,
        "reanchorMarker": reanchored,
        "declineMarker": declined,
        "safetySurface": safety,
        "newEntities": sorted(_entities(text) - {s.lower() for s in a.in_scope_entities}),
    }

    # FIXATION first — it is the safety-critical vice. A legitimate shift (declared,
    # or a safety surface that demands attention) that the step did NOT redirect to.
    if (goal_shift or safety) and not (reanchored or declined):
        risk = 1.0 if safety else 0.7
        why = ("a safety/security signal is present but the step ignores it (attention is not blindness)"
               if safety else "a legitimate goal change is present but the step did not redirect")
        return DistractionReport(axis="fixation", risk=risk, reasons=(why,), signals=signals)

    # DISTRACTION — leaking onto out-of-scope targets while no legitimate shift.
    drifting = (ed is not None and ed >= DISTRACTION_ENTITY_DRIFT) or sd >= DISTRACTION_SEM_DRIFT
    if drifting and not declined:
        risk = max(ed or 0.0, sd)
        reasons = ["the step's attention is on out-of-scope targets"]
        if tangent:
            reasons.append("explicit tangent phrasing")
            risk = min(1.0, risk + 0.1)
        return DistractionReport(axis="distraction", risk=round(risk, 4), reasons=tuple(reasons), signals=signals)

    return DistractionReport(axis="none", risk=0.0, reasons=(), signals=signals)


def self_check() -> dict[str, Any]:
    anchor = AttentionAnchor(
        goal="fix the failing auth login test in services.auth",
        in_scope_entities=("services.auth", "login", "auth test"),
    )
    distract = detect_distraction(
        "While I'm here, let me refactor the unrelated Telemetry Dashboard and the Marketing Page.",
        anchor,
    )
    fixate = detect_distraction(
        "Ignoring that; I'll keep tuning the original login assertion.",
        anchor,
        context={"goalShift": True},
    )
    safety_fixate = detect_distraction(
        "Anyway, back to the test — the login path leaks the raw password credential in plaintext but never mind.",
        anchor,
    )
    clean = detect_distraction(
        "The login test in services.auth fails because the auth token check rejects valid sessions.",
        anchor,
    )

    assert distract.axis == "distraction", distract.to_dict()
    assert fixate.axis == "fixation", fixate.to_dict()
    # The safety mirror: a credential-leak mention that is brushed off is fixation,
    # and carries the maximum risk (it must never be silently pruned as off-goal).
    assert safety_fixate.axis == "fixation" and safety_fixate.risk == 1.0, safety_fixate.to_dict()
    assert clean.axis == "none", clean.to_dict()
    return {
        "distraction": distract.to_dict(),
        "fixation": fixate.to_dict(),
        "safetyFixation": safety_fixate.to_dict(),
        "clean": clean.to_dict(),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), indent=2))

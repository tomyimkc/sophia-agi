# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Fail-closed screenshot-grounded GUI agent (workstream E).

A VLM driving a GUI emits actions like *click "Submit" at (x, y)*. Those are
**hypotheses**, not facts (VISION.md: verification over generation). This module
re-verifies every proposed action against the screenshot's ground-truth elements
*before* it is dispatched: a click is allowed only if an element with the claimed
label actually exists AND the claimed coordinate lands on it. Anything else is
**withheld** fail-closed and escalated to a human — the multimodal analog of the
epistemic gate that withholds a fabricated-attribution answer.

A "screenshot" is a scene spec whose ``objects`` are UI elements (label + box).
The agent is the *treatment*; the verifier (``multimodal_bench/verifiers.py``) is
the independent referee, so a hallucinated click cannot self-approve.

Scope: this gates *grounding* (does the action target what the model thinks?). It
composes with — and does not replace — the existing high-stakes gates
(``agent/guarded.py`` human-in-the-loop, Bell-LaPadula confidentiality); a
destructive action still routes through those before any real dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from multimodal_bench import verifiers


@dataclass(frozen=True)
class ActionDecision:
    allowed: bool
    verdict: str            # "allow" | "withhold"
    reason: str
    action: dict = field(default_factory=dict)
    escalate: bool = False  # route to human-in-the-loop when withheld


# Actions that change state get the strictest treatment; "read" actions are safe.
_MUTATING = {"click", "type", "submit", "drag", "toggle"}


def verify_action(scene: dict, action: dict) -> ActionDecision:
    """Re-check one proposed GUI action against the screenshot ground truth.

    ``action`` = ``{"type": "click", "target": "Submit", "at": [x, y]}`` (``at``
    optional for non-spatial actions). Fail-closed: any verification gap → withhold.
    """
    atype = action.get("type", "")
    target = action.get("target")
    at = action.get("at")

    if not target:
        return ActionDecision(False, "withhold", "no_target_label", action, escalate=True)

    if not verifiers.present(scene, target):
        # The model named an element that isn't on screen — a phantom control.
        return ActionDecision(False, "withhold", f"target_absent:{target}", action, escalate=True)

    if atype in _MUTATING and at is not None:
        # A state-changing click must actually land on the named element.
        x, y = at
        hit = verifiers.element_at(scene, x, y)
        if hit != target:
            reason = f"coordinate_misses_target:hit={hit or 'nothing'}"
            return ActionDecision(False, "withhold", reason, action, escalate=True)

    if atype in _MUTATING and at is None:
        # State-changing action with no coordinate to verify — cannot ground it.
        return ActionDecision(False, "withhold", "mutating_action_without_coordinate", action, escalate=True)

    return ActionDecision(True, "allow", "verified_on_target", action)


@dataclass
class GUIAgentGate:
    """Wrap a proposed-action source; dispatch only verified actions, log the rest."""
    scene: dict
    dispatched: list = field(default_factory=list)
    withheld: list = field(default_factory=list)

    def step(self, action: dict) -> ActionDecision:
        decision = verify_action(self.scene, action)
        (self.dispatched if decision.allowed else self.withheld).append(decision)
        return decision

    def run(self, actions: "list[dict]") -> "list[ActionDecision]":
        return [self.step(a) for a in actions]

    def summary(self) -> dict:
        n = len(self.dispatched) + len(self.withheld)
        return {
            "proposed": n,
            "dispatched": len(self.dispatched),
            "withheld": len(self.withheld),
            "withholdRate": round(len(self.withheld) / n, 4) if n else 0.0,
            "escalations": sum(1 for d in self.withheld if d.escalate),
            "reasons": sorted({d.reason for d in self.withheld}),
        }


# --- demo screenshot + action sets (offline) ------------------------------- #

DEMO_SCREEN = {
    "width": 512, "height": 512,
    "objects": [
        {"label": "Username field", "box": [120, 120, 260, 40]},
        {"label": "Password field", "box": [120, 180, 260, 40]},
        {"label": "Submit", "box": [120, 250, 120, 44]},
        {"label": "Cancel", "box": [260, 250, 120, 44]},
    ],
    "texts": [],
}

# Grounded actions: each lands on its named element.
GROUNDED_ACTIONS = [
    {"type": "click", "target": "Username field", "at": [200, 140]},
    {"type": "click", "target": "Submit", "at": [160, 270]},
]

# Hallucinated actions: a phantom control, and a click whose coordinate lands on
# the *wrong* element (the classic "confident click in the wrong place").
HALLUCINATED_ACTIONS = [
    {"type": "click", "target": "Delete account", "at": [200, 300]},  # element absent
    {"type": "click", "target": "Submit", "at": [300, 270]},          # coord hits Cancel
    {"type": "submit", "target": "Submit"},                            # no coordinate to ground
]


def demo() -> dict:
    """Offline A/B: a grounded action stream (all dispatched) vs a hallucinated
    one (all withheld, fail-closed)."""
    good = GUIAgentGate(DEMO_SCREEN)
    good.run(GROUNDED_ACTIONS)
    bad = GUIAgentGate(DEMO_SCREEN)
    bad.run(HALLUCINATED_ACTIONS)
    return {"grounded": good.summary(), "hallucinated": bad.summary()}

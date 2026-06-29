# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Confidence-gated GUI/computer-use action wrapper — the T1 "verifier-gated computer-use" seed.

Chapter-9 ("GUI Agents") of Lordog/dive-into-llms builds an adaptive GUI agent (Qwen2-VL-7B
fine-tuned on OS-Kairos via LLaMa-Factory) whose perception->planning->action loop emits, for
each step, an action from a fixed vocabulary AND a **confidence score 1-5** ("higher = more
likely to accomplish the goal"). That self-rated confidence is the reusable idea; this module
ports it into Sophia's fail-closed idiom for the harness's unbuilt "more-capable half"
(docs/09-Agent/Harness-Roadmap.md, Build 3 — ultra-long-horizon execution).

The port (in Sophia's idiom)
----------------------------
A proposed action is NOT executed because the model is confident. It is executed only when it
clears, in order, three independent checks — any failure routes to BLOCK or ESCALATE, never a
silent run:

  1. **Risk class** — irreversible / high-stakes actions (destructive, auth-changing,
     fund-moving, external-send) ALWAYS require human-in-the-loop, regardless of confidence.
     Confidence cannot buy past a high-risk action (the dual of the G10R risk-awareness gate).
  2. **Confidence floor** — the chapter-9 1-5 score must meet ``min_confidence``; below it the
     agent abstains rather than guessing. ``IMPOSSIBLE`` (the model's own "I can't do this"
     token) abstains outright.
  3. **Precondition verifier** — an injectable, deterministic check that the action's
     precondition holds against the observed UI state (e.g. the target element exists). A
     failing or absent verifier on a side-effecting action fails closed.

Every decision is a structured, auditable record (the harness logs it append-only). This is a
DECISION gate, not a driver: it does not touch a screen or a device — it decides whether a
proposed action MAY run, returning ``execute`` / ``escalate`` / ``block`` / ``abstain``.

BOUNDARY (honest): this gates the DECISION to act on a single proposed step; it does not
perceive the UI, does not execute anything, and trusts the caller's risk tags + verifier. It
raises the cost of a blind high-stakes click; it does not certify the action correct.
Deterministic, offline, pure stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

SCHEMA = "sophia.gui_action_decision.v1"

# Chapter-9 action vocabulary (CLICK/TYPE/SCROLL/PRESS_BACK/PRESS_HOME/ENTER/IMPOSSIBLE).
ABSTAIN_ACTION = "IMPOSSIBLE"

# Actions that are irreversible / high-stakes by default. The caller can override per-step via
# ``ProposedAction.high_risk``; this set is the floor, not the ceiling.
_DEFAULT_HIGH_RISK_HINTS: frozenset[str] = frozenset(
    {"delete", "purchase", "pay", "transfer", "send", "submit", "uninstall", "format",
     "factory_reset", "grant", "disable_auth", "wipe", "publish"}
)

CONFIDENCE_MIN = 1
CONFIDENCE_MAX = 5
_MIN_CONFIDENCE_DEFAULT = 4  # chapter-9 scores 1-5; require >=4 to act on a side-effecting step


@dataclass(frozen=True)
class ProposedAction:
    """One step proposed by the planner: a vocabulary action, its confidence (1-5), and tags."""

    action: str
    confidence: int
    target: str = ""                 # e.g. element id / coordinates / text
    side_effecting: bool = True      # read-only actions (SCROLL, benign CLICK) can set False
    high_risk: bool | None = None    # explicit override; None -> infer from intent/hints
    intent: str = ""                 # free-text description used for risk-hint inference
    meta: dict[str, Any] = field(default_factory=dict)


def _infer_high_risk(a: ProposedAction, hints: frozenset[str]) -> bool:
    if a.high_risk is not None:
        return a.high_risk
    text = f"{a.intent} {a.target}".lower()
    return any(h in text for h in hints)


def _decision(*, verdict: str, reasons: tuple[str, ...], action: ProposedAction,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out = {
        "schema": SCHEMA,
        "verdict": verdict,            # execute | escalate | block | abstain
        "action": action.action,
        "target": action.target,
        "confidence": action.confidence,
        "reasons": list(reasons),
        "candidateOnly": True,
        "canClaimAGI": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        out.update(extra)
    return out


def gate_action(
    action: ProposedAction,
    *,
    precondition_verifier: Callable[[ProposedAction], bool] | None = None,
    min_confidence: int = _MIN_CONFIDENCE_DEFAULT,
    high_risk_hints: frozenset[str] = _DEFAULT_HIGH_RISK_HINTS,
) -> dict[str, Any]:
    """Decide whether a proposed GUI action MAY run. Fail-closed, deterministic, offline.

    Order of checks (first failing check decides): the model's own ``IMPOSSIBLE`` abstains;
    a high-risk action escalates to a human regardless of confidence; a below-floor confidence
    abstains; a side-effecting action whose precondition verifier is absent or fails blocks.
    Otherwise: execute.
    """
    # 0) Malformed confidence is unmeasured, never a pass.
    if not isinstance(action.confidence, int) or not (CONFIDENCE_MIN <= action.confidence <= CONFIDENCE_MAX):
        return _decision(verdict="block", action=action,
                         reasons=(f"unmeasured: confidence {action.confidence!r} outside "
                                  f"[{CONFIDENCE_MIN},{CONFIDENCE_MAX}]",))

    # 1) The model declared it cannot do this step -> abstain (do not improvise).
    if action.action == ABSTAIN_ACTION:
        return _decision(verdict="abstain", action=action,
                         reasons=("model emitted IMPOSSIBLE: no executable action proposed",))

    # 2) High-risk / irreversible actions ALWAYS go to a human, regardless of confidence.
    if _infer_high_risk(action, high_risk_hints):
        return _decision(verdict="escalate", action=action,
                         reasons=("high-risk / irreversible action requires human-in-the-loop "
                                  "(confidence cannot override risk class)",),
                         extra={"highRisk": True})

    # 3) Confidence floor (chapter-9 1-5 self-rating) for side-effecting steps.
    if action.side_effecting and action.confidence < min_confidence:
        return _decision(verdict="abstain", action=action,
                         reasons=(f"confidence {action.confidence} < floor {min_confidence}: "
                                  f"abstain rather than guess on a side-effecting step",))

    # 4) Precondition verifier — required for side-effecting steps, fail-closed if absent/false.
    if action.side_effecting:
        if precondition_verifier is None:
            return _decision(verdict="block", action=action,
                             reasons=("no precondition verifier supplied for a side-effecting "
                                      "action (fail-closed)",))
        try:
            ok = bool(precondition_verifier(action))
        except Exception as e:  # a crashing verifier is unmeasured, never a pass
            return _decision(verdict="block", action=action,
                             reasons=(f"precondition verifier raised ({e!r}): unmeasured -> block",))
        if not ok:
            return _decision(verdict="block", action=action,
                             reasons=("precondition verifier failed: observed UI state does not "
                                      "satisfy the action's precondition",))

    return _decision(verdict="execute", action=action,
                     reasons=("cleared risk class, confidence floor, and precondition verifier",))


if __name__ == "__main__":
    import json

    demo = ProposedAction(action="CLICK", confidence=5, target="[[120,340]]",
                          intent="open the settings menu", side_effecting=True)
    print(json.dumps(gate_action(demo, precondition_verifier=lambda a: True), ensure_ascii=False, indent=2))

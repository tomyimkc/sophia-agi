# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Skill: sentinel_watch — classify a step + flag verdict drift (fail-closed).

The friendly surface over :mod:`agent.sentinel`. Given one harness step event and
a recent-vs-baseline verdict window, it returns a supervision verdict:
``review`` when the step failed on the gate or the verdict distribution drifted,
else ``ok``. Like every Sophia skill it never raises — on any error it abstains
(``verdict: "held"``) via the ``@sophia_skill`` wrapper.
"""
from __future__ import annotations

from agent.sentinel import classify_failure, detect_drift
from skills.core import sophia_skill


@sophia_skill(
    "sentinel_watch",
    summary="Classify a harness step + flag gate/verdict drift; returns ok/review (fail-closed).",
    uses=(),
)
def sentinel_watch(*, step: dict, recent: dict | None = None,
                   baseline: dict | None = None, threshold: float = 0.15) -> dict:
    fc = classify_failure(step) if isinstance(step, dict) else None
    drift = detect_drift(recent or {}, baseline or {}, threshold=threshold) if baseline else {"drift": False}
    flagged = bool(fc and fc in ("gate_violation", "verifier_fail")) or bool(drift.get("drift"))
    return {
        "verdict": "review" if flagged else "ok",
        "failureClass": fc,
        "drift": drift,
    }

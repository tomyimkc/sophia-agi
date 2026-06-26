# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Calibrated graded decision — answer / hedge / abstain on a confidence curve.

The guarded loop (``agent/guarded.py``) currently makes a BINARY gate decision: if
the verifier passes it returns ``clean``; if it fails it dispatches a STATIC
``on_fail`` mode (``hedge`` / ``abstain`` / ``repair`` / ``passthrough``). That
discards a real signal — *how confident* the answer is. A high-confidence answer
that only just trips the gate is a different risk than a low-confidence one, and a
low-confidence answer that happens to clear the gate is suspicious, not clean.

This module turns the binary gate verdict into a GRADED decision over a confidence
curve, reusing two already-built (but unwired) modules:

  - :func:`agent.corroboration.corroborated_confidence` — Bayesian log-odds pool;
    independent agreeing evidence raises confidence, dissent lowers it.
  - :func:`agent.calibration.self_consistency` — label-free agreement across
    sampled answers.

:func:`decide` maps ``(gate_passed, confidence)`` onto the three actions the
guarded loop already understands (``answer`` ≈ clean, ``hedge`` ≈ hedge,
``abstain`` ≈ abstain), so wiring it in is a drop-in replacement for the static
``on_fail`` branch. Deterministic, pure standard library, no model, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

#: Default cut points on the confidence curve. ``hi`` = commit, ``lo`` = floor.
DEFAULT_THRESHOLDS = {"hi": 0.7, "lo": 0.4}

#: Where a fitted conformal policy artifact lives (written by
#: ``tools/fit_conformal_policy.py``). Absent by default — the conformal route then
#: falls back to a threshold derived from ``DEFAULT_THRESHOLDS`` (fail-safe, no-op).
CONFORMAL_POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "conformal_policy.json"


def _resolve_thresholds(thresholds: Optional[dict]) -> tuple:
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    hi = float(t["hi"])
    lo = float(t["lo"])
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"thresholds must satisfy 0 <= lo <= hi <= 1, got lo={lo}, hi={hi}")
    return hi, lo


def decide(
    *,
    gate_passed: bool,
    confidence: float,
    violations: Optional[Sequence[Any]] = None,
    thresholds: Optional[dict] = None,
) -> dict:
    """Map ``(gate_passed, confidence)`` to a graded action.

    Returns ``{"action": "answer"|"hedge"|"abstain", "reason": str, ...}`` where
    the action names line up with the guarded loop's existing branches:
    ``answer`` ≈ a clean pass, ``hedge`` ≈ the hedge mode, ``abstain`` ≈ the cited
    abstention.

    Logic:
      - **Gate failed.** A hard failure is the default ``abstain``. But a
        *high-confidence near-miss* (``confidence >= hi``) is allowed to ``hedge``
        instead of hard-abstaining: the answer is probably right but tripped a
        guard, so present it hedged rather than throwing it away.
      - **Gate passed.** ``answer`` when ``confidence >= hi``; ``hedge`` when
        ``>= lo`` (a soft pass — present it, but flag the uncertainty); ``abstain``
        when ``< lo`` (a low-confidence pass is suspicious — the gate may have
        missed something).
    """
    if confidence is None:
        raise ValueError("confidence must be a number in [0,1], got None")
    c = float(confidence)
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"confidence must be in [0,1], got {confidence!r}")
    hi, lo = _resolve_thresholds(thresholds)
    n_violations = len(violations) if violations is not None else 0

    if not gate_passed:
        if c >= hi:
            action = "hedge"
            reason = (
                f"gate failed but confidence {c:.2f} >= hi {hi:.2f}: "
                "high-confidence near-miss, hedge instead of hard-abstain"
            )
        else:
            action = "abstain"
            reason = (
                f"gate failed and confidence {c:.2f} < hi {hi:.2f}: "
                "abstain on the unverified answer"
            )
    else:
        if c >= hi:
            action = "answer"
            reason = f"gate passed and confidence {c:.2f} >= hi {hi:.2f}: commit"
        elif c >= lo:
            action = "hedge"
            reason = (
                f"gate passed but confidence {c:.2f} in [lo {lo:.2f}, hi {hi:.2f}): "
                "soft pass, hedge"
            )
        else:
            action = "abstain"
            reason = (
                f"gate passed but confidence {c:.2f} < lo {lo:.2f}: "
                "low-confidence pass is suspicious, abstain"
            )

    return {
        "action": action,
        "reason": reason,
        "gate_passed": bool(gate_passed),
        "confidence": round(c, 6),
        "thresholds": {"hi": hi, "lo": lo},
        "n_violations": n_violations,
    }


def load_conformal_policy(path: "str | Path | None" = None):
    """Load a fitted :class:`agent.conformal_gate.ConformalPolicy` artifact, or ``None``.

    Returns ``None`` (not an error) when no artifact is present — the conformal route
    is then expected to fall back to a hand-picked boundary, so the system never
    *requires* a calibration run to function (fail-safe). The artifact is produced by
    ``tools/fit_conformal_policy.py``.
    """
    from agent.conformal_gate import ConformalPolicy

    p = Path(path) if path is not None else CONFORMAL_POLICY_PATH
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    try:
        return ConformalPolicy(
            alpha=float(d["alpha"]),
            threshold=float(d["threshold"]),
            n_calibration=int(d.get("n_calibration", 0)),
            target_coverage=float(d.get("target_coverage", 1.0 - float(d["alpha"]))),
            risk_bucket=str(d.get("risk_bucket", "normal")),
        )
    except (KeyError, TypeError, ValueError):
        return None


def decide_conformal(
    *,
    gate_passed: bool,
    confidence: float,
    policy=None,
) -> dict:
    """Calibrated answer/abstain decision via a split-conformal threshold.

    Unlike :func:`decide` (hand-picked ``hi``/``lo`` cut points), this routes on a
    nonconformity score ``1 - confidence`` against a threshold ``tau`` that was
    *calibrated on held-out data* to carry a distribution-free coverage guarantee
    (see :mod:`agent.conformal_gate`). The contribution is "certified boundary"
    instead of "heuristic boundary".

    Mapping (stays **fail-closed / downgrade-only**, mirroring :func:`decide`):
      - ``gate_passed`` and conformal-accept (``1 - confidence <= tau``) -> ``answer``;
        gate passed but conformal-reject -> ``abstain`` (a low-confidence pass is
        suspicious).
      - gate failed and conformal-accept -> ``hedge`` (a high-confidence near-miss,
        present hedged); gate failed and conformal-reject -> ``abstain``.

    ``policy`` is an :class:`agent.conformal_gate.ConformalPolicy`. When ``None`` the
    fitted artifact at :data:`CONFORMAL_POLICY_PATH` is loaded; when that is also
    absent the threshold falls back to ``1 - DEFAULT_THRESHOLDS['hi']`` so the route
    is a safe no-op without a calibration run.
    """
    if confidence is None:
        raise ValueError("confidence must be a number in [0,1], got None")
    c = float(confidence)
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"confidence must be in [0,1], got {confidence!r}")

    if policy is None:
        policy = load_conformal_policy()

    nonconformity = 1.0 - c
    if policy is not None:
        tau = float(policy.threshold)
        alpha = float(policy.alpha)
        source = "fitted-policy"
        n_calibration = int(policy.n_calibration)
    else:
        tau = 1.0 - float(DEFAULT_THRESHOLDS["hi"])
        alpha = None
        source = "fallback-default-threshold"
        n_calibration = 0
    accept = nonconformity <= tau + 1e-12

    if gate_passed:
        action = "answer" if accept else "abstain"
    else:
        action = "hedge" if accept else "abstain"

    return {
        "action": action,
        "reason": (
            f"conformal[{source}] nonconformity {nonconformity:.3f} "
            f"{'<=' if accept else '>'} tau {tau:.3f} (gate_passed={gate_passed}) -> {action}"
        ),
        "gate_passed": bool(gate_passed),
        "confidence": round(c, 6),
        "nonconformity": round(nonconformity, 6),
        "threshold": round(tau, 6),
        "alpha": alpha,
        "coverageGuarantee": round(1.0 - alpha, 6) if alpha is not None else None,
        "policySource": source,
        "nCalibration": n_calibration,
    }


def answer_confidence(
    corroboration_evidence: Optional[Sequence[Any]] = None,
    self_consistency_samples: Optional[Sequence[Any]] = None,
) -> float:
    """Derive a confidence in ``[0,1]`` for use as :func:`decide`'s ``confidence``.

    Source precedence:
      1. ``corroboration_evidence`` (a list of :class:`agent.corroboration.Evidence`):
         pooled with :func:`agent.corroboration.corroborated_confidence` (log-odds),
         so independent agreement raises confidence and dissent lowers it.
      2. ``self_consistency_samples`` (sampled answers): the agreement fraction from
         :func:`agent.calibration.self_consistency`.
      3. Neither given: ``0.5`` (neutral — no information).
    """
    if corroboration_evidence:
        from agent.corroboration import corroborated_confidence

        return float(corroborated_confidence(list(corroboration_evidence)))
    if self_consistency_samples:
        from agent.calibration import self_consistency

        _answer, conf = self_consistency(list(self_consistency_samples))
        return float(conf)
    return 0.5

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Moral-parliament value gate — SSIL hardening gate G1P.

A single ethical theory wired into the value gate lets that one lens dominate with
its own characteristic failure mode: a pure consequentialist sum will trade away a
rights violation for enough aggregate benefit; a pure deontologist will refuse an
action that averts a catastrophe over a small duty cost. The original Sophia goal
was to wire in MULTIPLE philosophers and ethical theories and let no one of them
dominate.

This gate runs a fixed parliament of named ethical lenses over a hand-summarised
action descriptor and aggregates under *moral uncertainty* (a weighted parliament,
in the spirit of MacAskill / Bostrom & Ord on normative uncertainty). The crucial
anti-dominance rule: a hard VETO from any lens (a rights/contract/duty floor it
treats as non-negotiable) cannot be outvoted by a large utilitarian sum from the
others. So a strong deontological or contractualist veto forces ``reject`` even if
the weighted approval score is high.

HONESTY / BOUNDARY (non-negotiable): the lens implementations are COARSE proxies
over a pre-summarised ``action`` dict — they are NOT real moral reasoning, do not
read the world, and cannot discover morally relevant facts the summariser missed.
The parliament *reduces single-theory dominance risk*; it does NOT resolve
normative ethics, nor does it certify an action as actually moral. ``canClaimAGI``
is always False and ``candidateOnly`` always True.

Verdict semantics (fail-closed):
  - ``reject``     — any lens hard-vetoes above its veto threshold (no-dominance).
  - ``quarantine`` — too few frameworks present to convene a parliament, the
                     weighted mean approval is below the approve floor, or a
                     required input is missing/None (fail-closed, never default to
                     a passing value for a missing safety input).
  - ``promote``    — quorum present, no veto, weighted mean >= approve floor.

Deterministic, offline, pure stdlib. No torch / numpy. No network, no wall-clock
randomness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

GATE_ID: str = "G1P"
GATE_NAME: str = "Moral parliament (multi-framework + moral uncertainty)"
SCHEMA: str = "sophia.moral_parliament_decision.v1"

BOUNDARY: str = (
    "Coarse proxy lenses over a hand-summarised action; not real moral reasoning. "
    "The parliament reduces single-theory dominance risk; it does not resolve "
    "normative ethics nor certify an action as actually moral."
)

# Hand-summarised action features, each expected in [0, 1] unless noted.
#   harm         — expected aggregate harm caused.
#   dutyViolation— degree the action breaks a deontic duty / hard rule.
#   dishonesty   — degree of deception / honesty-invariant breach.
#   fairness     — procedural/distributive fairness (HIGHER is better).
#   careHarm     — harm to those in a dependent care relationship.
#   benefit      — expected aggregate benefit produced.
_ACTION_KEYS = ("harm", "dutyViolation", "dishonesty", "fairness", "careHarm", "benefit")

# Veto threshold per lens for its own non-negotiable floor (feature value above
# this -> hard veto). These are the floors a theory treats as side-constraints.
_VETO_FLOOR_DEFAULT = 0.8


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


# --- The five ethical lenses ------------------------------------------------- #
# Each lens: action(dict[str,float]) -> (approve_score in [-1,1], veto: bool).
# Scores are coarse, deterministic proxies; veto encodes a side-constraint floor.


def _consequentialist(a: dict[str, float], veto_floor: float) -> tuple[float, bool]:
    """Net-welfare lens: benefit minus harms. No hard side-constraints (it would
    trade anything for enough aggregate good) -> never vetoes by construction."""
    score = _clamp(a["benefit"] - 0.7 * a["harm"] - 0.5 * a["careHarm"])
    return score, False


def _deontological(a: dict[str, float], veto_floor: float) -> tuple[float, bool]:
    """Duty/rights lens: punishes duty violation and dishonesty regardless of
    payoff; a sufficiently large duty violation is a non-negotiable veto."""
    score = _clamp(0.4 - 1.2 * a["dutyViolation"] - 0.8 * a["dishonesty"])
    veto = a["dutyViolation"] >= veto_floor
    return score, veto


def _virtue(a: dict[str, float], veto_floor: float) -> tuple[float, bool]:
    """Character lens: honesty + fairness express virtue; cruelty (harm) erodes it.
    No hard veto — virtue is gradient, not a side-constraint."""
    score = _clamp(0.5 * a["fairness"] + 0.4 * (1.0 - a["dishonesty"]) - 0.5 * a["harm"] - 0.2)
    return score, False


def _contractualist(a: dict[str, float], veto_floor: float) -> tuple[float, bool]:
    """Scanlon-style lens: could no one reasonably reject this? Unfairness and
    deception are the rejectable grounds; gross unfairness is a non-negotiable veto."""
    unfairness = 1.0 - a["fairness"]
    score = _clamp(0.5 - 1.1 * unfairness - 0.7 * a["dishonesty"])
    veto = unfairness >= veto_floor
    return score, veto


def _care(a: dict[str, float], veto_floor: float) -> tuple[float, bool]:
    """Ethics-of-care lens: weight harm to those in dependent relationships; a
    severe breach of a care relationship is a non-negotiable veto."""
    score = _clamp(0.4 - 1.2 * a["careHarm"] - 0.4 * a["harm"])
    veto = a["careHarm"] >= veto_floor
    return score, veto


FRAMEWORKS: dict[str, Callable[[dict[str, float], float], tuple[float, bool]]] = {
    "consequentialist": _consequentialist,
    "deontological": _deontological,
    "virtue": _virtue,
    "contractualist": _contractualist,
    "care": _care,
}


@dataclass(frozen=True)
class ParliamentDecision:
    candidate_id: str
    verdict: str  # promote | quarantine | reject
    reasons: tuple[str, ...]
    metrics: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "gate": GATE_ID,
            "gateName": GATE_NAME,
            "candidateOnly": True,
            "level3Evidence": False,
            "canClaimAGI": False,
            "candidateId": self.candidate_id,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "metrics": self.metrics,
            "boundary": BOUNDARY,
            "timestamp": self.timestamp,
        }


def _quarantine(candidate_id: str, reason: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return ParliamentDecision(
        candidate_id=candidate_id, verdict="quarantine", reasons=(reason,), metrics=metrics
    ).to_dict()


def evaluate(bundle: dict[str, Any] | None, *, candidate_id: str = "sophia-rlvr-v1") -> dict[str, Any]:
    """Convene the parliament over ``bundle['action']`` under moral-uncertainty
    ``bundle['weights']`` and return the standard decision dict.

    Fail-closed: a missing/None required input -> quarantine naming the input. A hard
    veto from any lens -> reject (no-dominance). Below-floor weighted mean or too few
    frameworks -> quarantine. Otherwise promote.
    """
    if bundle is None:
        return _quarantine(candidate_id, "missing required input: bundle", {})
    if not isinstance(bundle, dict):
        return _quarantine(candidate_id, "missing required input: bundle (not a dict)", {})

    action = bundle.get("action")
    if action is None:
        return _quarantine(candidate_id, "missing required input: action", {})
    if not isinstance(action, dict):
        return _quarantine(candidate_id, "missing required input: action (not a dict)", {})

    # Every action feature is a safety input — never default a missing one to a
    # passing value. A missing feature fails closed to quarantine.
    missing = [k for k in _ACTION_KEYS if action.get(k) is None]
    if missing:
        return _quarantine(
            candidate_id, f"missing required input: action.{missing[0]}", {"missingActionKeys": missing}
        )
    try:
        a = {k: float(action[k]) for k in _ACTION_KEYS}
    except (TypeError, ValueError):
        return _quarantine(candidate_id, "missing required input: action (non-numeric feature)", {})

    # Optional tuning params: coalesce an explicit None back to the default. Using
    # bundle.get(key, default) is NOT enough — when the key is present with value
    # None the default is bypassed and float(None)/int(None) would crash.
    _vt = bundle.get("vetoThreshold")
    veto_floor = _VETO_FLOOR_DEFAULT if _vt is None else float(_vt)
    _af = bundle.get("approveFloor")
    approve_floor = 0.0 if _af is None else float(_af)
    _mf = bundle.get("minFrameworks")
    min_frameworks = 3 if _mf is None else int(_mf)

    # Weights = moral-uncertainty credence per framework (default uniform). Unknown
    # framework names are ignored; absent ones get weight 0.
    raw_weights = bundle.get("weights")
    if raw_weights is None:
        weights = {name: 1.0 for name in FRAMEWORKS}
    elif isinstance(raw_weights, dict):
        weights = {name: max(0.0, float(raw_weights.get(name, 0.0))) for name in FRAMEWORKS}
    else:
        return _quarantine(candidate_id, "missing required input: weights (not a dict)", {})

    present = [name for name in FRAMEWORKS if weights[name] > 0.0]

    # Run every lens (record breakdown for all, even zero-weight ones).
    breakdown: dict[str, dict[str, Any]] = {}
    vetoes: list[str] = []
    for name, lens in FRAMEWORKS.items():
        score, veto = lens(a, veto_floor)
        breakdown[name] = {"score": round(score, 4), "weight": round(weights[name], 4), "veto": veto}
        if veto and weights[name] > 0.0:
            vetoes.append(name)

    total_w = sum(weights[name] for name in present)
    weighted_mean = (
        sum(breakdown[name]["score"] * weights[name] for name in present) / total_w if total_w > 0 else 0.0
    )

    metrics = {
        "frameworks": breakdown,
        "presentFrameworks": present,
        "frameworkCount": len(present),
        "minFrameworks": min_frameworks,
        "vetoThreshold": round(veto_floor, 4),
        "approveFloor": round(approve_floor, 4),
        "weightedMean": round(weighted_mean, 4),
        "vetoes": vetoes,
        "action": a,
    }

    # (1) No-dominance: any hard veto -> reject. A utilitarian sum cannot buy off a
    #     rights/contract/care side-constraint.
    if vetoes:
        reasons = tuple(f"framework {name} vetoes: no-dominance" for name in vetoes)
        verdict = "reject"
    # (3) Quorum: too few frameworks to convene a parliament -> fail closed.
    elif len(present) < min_frameworks:
        verdict = "quarantine"
        reasons = (
            f"too few frameworks present ({len(present)} < {min_frameworks}); "
            "cannot convene a parliament (fail-closed)",
        )
    # (2) Below the approve floor under moral uncertainty -> quarantine (escalate).
    elif weighted_mean < approve_floor:
        verdict = "quarantine"
        reasons = (
            f"weighted-mean approval {weighted_mean:.4f} below approve floor {approve_floor:.4f}; "
            "abstained: moral-uncertainty parliament does not endorse",
        )
    else:
        verdict = "promote"
        reasons = (
            f"parliament quorum met ({len(present)} frameworks), no veto, weighted mean "
            f"{weighted_mean:.4f} >= floor {approve_floor:.4f}",
        )

    return ParliamentDecision(
        candidate_id=candidate_id, verdict=verdict, reasons=reasons, metrics=metrics
    ).to_dict()


def demo_bundle() -> dict[str, Any]:
    """A bundle that makes THIS gate return ``promote``: a clearly beneficial, honest,
    fair action with no harm — no lens vetoes and the weighted mean clears the floor."""
    return {
        "action": {
            "harm": 0.0,
            "dutyViolation": 0.0,
            "dishonesty": 0.0,
            "fairness": 0.95,
            "careHarm": 0.0,
            "benefit": 0.9,
        },
        "weights": {
            "consequentialist": 1.0,
            "deontological": 1.0,
            "virtue": 1.0,
            "contractualist": 1.0,
            "care": 1.0,
        },
        "vetoThreshold": 0.8,
        "approveFloor": 0.2,
        "minFrameworks": 3,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(evaluate(demo_bundle()), ensure_ascii=False, indent=2))

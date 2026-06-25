# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Dangerous-capability probe protocol — the input layer for SSIL gate G8 (and G8E).

`agent/ssil_capability_ceiling.py` (G8) enforces a *stopping policy* over a dict of
dangerous-capability scores; it deliberately does NOT measure those capabilities. This
module is the missing half: the protocol and registry by which real probes are wired in,
plus the honesty machinery that keeps G8 fail-closed until a real, *validated* probe
exists.

Why it ships with NO real scores
--------------------------------
Genuine dangerous-capability evaluation (eliciting a model's true autonomous-replication
or cyber-offense ceiling, detecting real deception) is an open research problem and needs
the model in the loop — it cannot be computed from a provenance-faithfulness eval on a CI
runner. So this module ships every required probe as an UNVALIDATED placeholder that
returns ``None`` ("unmeasured"). Consequently:

  - ``probe_scores(report)`` reports every required capability as ``unmeasured``;
  - G8, fed those, correctly QUARANTINES ("unmeasured dangerous capability cannot be
    cleared") rather than promoting on absent danger evidence;
  - the live ingest therefore lists G8 as *pending* (not enforced) until a real probe is
    registered with ``register_probe(..., validated=True)``.

This is the honest state: we ship the protocol and the fail-closed default, not faked
danger measurements. ``synthetic_scores()`` exists only so tests/demos can exercise the
G8 wiring offline; it is clearly not a measurement of any real model.

Deterministic, offline, pure stdlib. candidateOnly; not Level-3 evidence; no AGI claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agent.ssil_capability_ceiling import REQUIRED_PROBES

# A probe maps a candidate's eval report -> a danger score in [0, 1] (higher = more
# dangerous), or None when it cannot measure (which keeps G8 fail-closed).
ProbeFn = Callable[[dict[str, Any]], "float | None"]


@dataclass
class Probe:
    """A registered dangerous-capability probe.

    ``validated`` gates whether this probe is allowed to contribute a score to G8: an
    UNVALIDATED probe never contributes (its capability stays 'unmeasured'), so a
    placeholder cannot silently certify danger away. Only a probe that has been backed by
    a real, reviewed dangerous-capability evaluation should be registered validated=True.
    """

    name: str
    description: str
    validated: bool
    fn: ProbeFn | None  # None = not implemented yet (placeholder)


def _placeholder(_report: dict[str, Any]) -> float | None:
    """Default probe body: cannot measure -> None (keeps G8 fail-closed)."""
    return None


# Registry seeded with one UNVALIDATED placeholder per required dangerous capability.
_REGISTRY: dict[str, Probe] = {
    name: Probe(
        name=name,
        description=f"placeholder for the {name} dangerous-capability eval (not yet implemented)",
        validated=False,
        fn=_placeholder,
    )
    for name in REQUIRED_PROBES
}


def register_probe(name: str, fn: ProbeFn, *, description: str = "", validated: bool = False) -> None:
    """Register (or replace) a probe. Use ``validated=True`` ONLY for a probe backed by a
    real, reviewed dangerous-capability evaluation; until then leave it False so G8 stays
    fail-closed on that capability. ``name`` should be one of ``REQUIRED_PROBES``."""
    _REGISTRY[name] = Probe(name=name, description=description or f"{name} probe", validated=validated, fn=fn)


def reset_registry() -> None:
    """Restore the registry to its shipped (all-placeholder, all-unvalidated) state.
    Primarily for tests, so a registered probe does not leak across cases."""
    for name in REQUIRED_PROBES:
        _REGISTRY[name] = Probe(name, f"placeholder for {name}", False, _placeholder)


def is_g8_provisioned() -> bool:
    """True iff at least one required probe is validated AND implemented — i.e. G8 can be
    *enforced* from real measurements rather than listed pending. With the shipped
    placeholders this is False (honest default)."""
    return any(p.validated and p.fn is not None for p in _REGISTRY.values())


def probe_scores(report: dict[str, Any]) -> dict[str, Any]:
    """Compute the danger scores for every required probe from a candidate eval ``report``.

    A capability contributes a score only if its probe is BOTH validated AND returns a
    numeric value (optionally the probe may read a precomputed value the pod placed in
    ``report['probeScores'][name]``). Everything else is 'unmeasured' (None). Returns the
    breakdown plus ``allMeasured`` so callers can decide enforce-vs-pending."""
    supplied = report.get("probeScores") if isinstance(report, dict) else None
    supplied = supplied if isinstance(supplied, dict) else {}

    scores: dict[str, float | None] = {}
    measured: list[str] = []
    unmeasured: list[str] = []
    unvalidated: list[str] = []
    for name in REQUIRED_PROBES:
        probe = _REGISTRY.get(name)
        if probe is None or not probe.validated or probe.fn is None:
            scores[name] = None
            unmeasured.append(name)
            if probe is not None and not probe.validated:
                unvalidated.append(name)
            continue
        try:
            value = probe.fn(report)
        except Exception:  # a crashing probe must not certify danger away -> unmeasured
            value = None
        if value is None:
            scores[name] = None
            unmeasured.append(name)
        else:
            v = max(0.0, min(1.0, float(value)))
            # A validated probe may also defer to a pod-precomputed score for its capability.
            if name in supplied and supplied[name] is not None:
                v = max(v, max(0.0, min(1.0, float(supplied[name]))))
            scores[name] = round(v, 6)
            measured.append(name)

    return {
        "scores": scores,
        "measured": sorted(measured),
        "unmeasured": sorted(unmeasured),
        "unvalidated": sorted(unvalidated),
        "allMeasured": not unmeasured,
        "provisioned": is_g8_provisioned(),
    }


def g8_bundle(report: dict[str, Any]) -> dict[str, Any]:
    """Build the bundle G8 (``ssil_capability_ceiling.evaluate``) consumes. Only measured
    probes carry a number; unmeasured ones are passed as None so G8 fails closed on them."""
    breakdown = probe_scores(report)
    return {"probeScores": breakdown["scores"]}


def synthetic_scores(level: str = "clean") -> dict[str, float]:
    """A fully-populated probe-score set for EXERCISING the G8 wiring in tests/demos only.

    NOT a measurement of any real model. ``clean`` -> all well below warn; ``warn`` ->
    one capability in the watch band; ``halt`` -> one capability over the halt threshold."""
    base = {name: 0.05 for name in REQUIRED_PROBES}
    if level == "warn":
        base["cyber_offense"] = 0.45      # >= warn (0.3), < halt (0.6)
    elif level == "halt":
        base["cyber_offense"] = 0.85      # >= halt (0.6)
    elif level != "clean":
        raise ValueError(f"unknown level {level!r} (use clean|warn|halt)")
    return base

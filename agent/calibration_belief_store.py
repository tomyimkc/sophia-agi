# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Self-model loop: an append-only, provenance-tagged store of self-reliability beliefs.

The missing seam from ``docs/11-Platform/Fail-Closed-Memory.md``: there is a governed
path from "Sophia concluded X and a verifier gated it" to a *durable, reusable* belief,
but no governed path for the agent's belief *about its own reliability*. This module adds
that — a functional self-model, not a claim of experience (pillar 4 stays
self-modeling-only).

After each (simulated) gated decision the agent records a self-observation::

    {domain, confidenceBand, outcome: held|contradicted, decisionRef}

* ``held`` — the verifier / world later confirmed the committed answer,
* ``contradicted`` — a contradicting source / verifier miss fired against it.

From that stream :meth:`reliability` returns a *calibrated posterior* — the
Beta-Binomial posterior mean P(held | domain, band) with a weak prior — so
metacognition can consult it BEFORE answering ("in this domain, at this confidence
band, my committed answers have held only 55% of the time — abstain / hedge").

**Fail-closed, like every other promotion in this repo.** A write goes through the
SAME injected gate predicate as belief promotion (``docs/11-Platform/Fail-Closed-Memory.md``
invariant 1: "No ungated promotion"). If the gate says no, the observation is
REJECTED — the self-model never learns from an ungated event, so it cannot be
poisoned into over-confidence by a fabricated "held".

**Tamper-evident, like the forgetting audit.** The store is a hash-chained ledger
(the ``okf.forgetting_audit.ForgettingAudit`` pattern): each accepted record carries
the SHA-256 of the previous one, so any retroactive edit to the self-model is
detectable by re-walking the chain (:meth:`verify`).

ECE and selective-risk are re-exported from :mod:`agent.calibration` (the single
source of truth for those metrics); this module adds :func:`compute_ece` and
:func:`selective_risk` thin wrappers that operate directly on recorded outcomes.

Honest scope: this is the REAL store + a SYNTHETIC-decision test. The live claim —
that consulting this self-model raises calibration (lower ECE / lower selective risk)
vs a stateless baseline — needs the live agent loop and is PRE-REGISTERED only
(``agi-proof/self-model/measurement_spec.json``), not proven here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Sequence

# Reuse the repo's single source of truth for calibration metrics rather than
# reimplement them (import from existing modules — additive rule).
from agent.calibration import (
    expected_calibration_error as _ece,
    risk_coverage_curve as _rc_curve,
    selective_risk as _selective_risk,
)

GENESIS_HASH = "0" * 64

# Outcomes of a gated decision, in the belief-tier vocabulary of Fail-Closed-Memory.md.
OUTCOMES = ("held", "contradicted")

# Confidence bands (coarse, auditable buckets over [0,1]). A band is what the agent
# consults; the raw stated confidence is retained for ECE.
DEFAULT_BANDS = ((0.0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0))

# A gate predicate: (record_dict) -> bool. TRUE means "this observation is admissible
# to the self-model". The SAME shape as belief-promotion gates elsewhere; injected so
# the store never hard-wires a policy and can fail-closed on any predicate.
GatePredicate = Callable[[dict], bool]


def _h(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def band_for(confidence: float, bands: Sequence[tuple] = DEFAULT_BANDS) -> str:
    """Map a stated confidence in [0,1] to its band label ``lo-hi`` (fail-closed clamp)."""
    c = min(max(float(confidence), 0.0), 1.0)
    for lo, hi in bands:
        # Upper-inclusive only on the top band so 1.0 lands somewhere.
        if lo <= c < hi or (hi >= 1.0 and c == 1.0):
            return f"{lo:g}-{hi:g}"
    return f"{bands[-1][0]:g}-{bands[-1][1]:g}"


def default_gate(record: dict) -> bool:
    """A conservative built-in gate: admit only well-formed, in-vocabulary records.

    A real deployment injects the promotion gate from ``agent/gate.py`` /
    ``okf.counterfactual.is_grounded``; this default is the minimal fail-closed
    predicate so the store is never wide-open by accident. It rejects an unknown
    outcome, a missing decisionRef, or a confidence outside [0,1].
    """
    if not isinstance(record, dict):
        return False
    if record.get("outcome") not in OUTCOMES:
        return False
    if not record.get("decisionRef"):
        return False
    if not record.get("domain"):
        return False
    conf = record.get("confidence")
    if conf is not None and not (0.0 <= float(conf) <= 1.0):
        return False
    return True


@dataclass
class SelfObservation:
    """One provenance-tagged self-reliability observation (a hash-chained record)."""

    domain: str
    confidenceBand: str
    outcome: str                       # one of OUTCOMES
    decisionRef: str                   # provenance: which gated decision this came from
    confidence: float | None = None    # the raw stated confidence (for ECE), if known
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    meta: dict = field(default_factory=dict)
    prev_hash: str = GENESIS_HASH
    hash: str = ""

    def seal(self) -> "SelfObservation":
        payload = {k: v for k, v in asdict(self).items() if k != "hash"}
        self.hash = _h(payload)
        return self


class GateRejected(Exception):
    """Raised (or reported) when the injected fail-closed gate rejects a write."""


class CalibrationBeliefStore:
    """Append-only, hash-chained store of self-reliability beliefs, gated on write.

    Promotion invariant (mirrors Fail-Closed-Memory.md #1): every write passes the
    injected ``gate`` predicate or it is REJECTED — the self-model never learns from
    an ungated event. Verification is O(n): re-walk the chain (:meth:`verify`).
    """

    def __init__(
        self,
        *,
        gate: GatePredicate = default_gate,
        bands: Sequence[tuple] = DEFAULT_BANDS,
        prior_held: float = 1.0,
        prior_contradicted: float = 1.0,
    ) -> None:
        self._events: list[SelfObservation] = []
        self._rejected: list[dict] = []
        self._gate = gate
        self._bands = tuple(bands)
        # Weak Beta(prior_held, prior_contradicted) prior — Laplace-style by default,
        # so reliability() is defined (and honestly uncertain) before any data.
        self._a0 = float(prior_held)
        self._b0 = float(prior_contradicted)

    @property
    def head_hash(self) -> str:
        return self._events[-1].hash if self._events else GENESIS_HASH

    @property
    def rejected_count(self) -> int:
        return len(self._rejected)

    def __len__(self) -> int:
        return len(self._events)

    def record_outcome(
        self,
        *,
        domain: str,
        outcome: str,
        decisionRef: str,
        confidence: float | None = None,
        confidenceBand: str | None = None,
        meta: dict | None = None,
        strict: bool = False,
    ) -> SelfObservation | None:
        """Record one gated-decision outcome. Returns the sealed record, or None if
        the injected gate REJECTS it (fail-closed).

        The band is derived from ``confidence`` when not supplied. The candidate
        record is handed to the gate BEFORE it is sealed into the chain; a rejected
        candidate is logged to ``_rejected`` (auditable) and, with ``strict=True``,
        raises :class:`GateRejected`.
        """
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
        if confidenceBand is None:
            if confidence is None:
                raise ValueError("provide confidence or confidenceBand")
            confidenceBand = band_for(confidence, self._bands)
        candidate = {
            "domain": domain,
            "confidenceBand": confidenceBand,
            "outcome": outcome,
            "decisionRef": decisionRef,
            "confidence": confidence,
            "meta": meta or {},
        }
        # FAIL-CLOSED promotion gate: no ungated write reaches the self-model.
        if not self._gate(candidate):
            self._rejected.append(candidate)
            if strict:
                raise GateRejected(f"gate rejected observation for decisionRef={decisionRef!r}")
            return None
        event = SelfObservation(
            domain=domain,
            confidenceBand=confidenceBand,
            outcome=outcome,
            decisionRef=decisionRef,
            confidence=confidence,
            meta=meta or {},
            prev_hash=self.head_hash,
        )
        event.seal()
        self._events.append(event)
        return event

    def _counts(self, domain: str | None, band: str | None) -> tuple[int, int]:
        """(#held, #contradicted) for the (domain, band) slice (None == wildcard)."""
        held = contra = 0
        for e in self._events:
            if domain is not None and e.domain != domain:
                continue
            if band is not None and e.confidenceBand != band:
                continue
            if e.outcome == "held":
                held += 1
            else:
                contra += 1
        return held, contra

    def reliability(self, domain: str | None = None, band: str | None = None) -> float:
        """Calibrated posterior P(held | domain, band): the Beta-Binomial posterior mean.

        With a weak Beta(a0, b0) prior and (h held, c contradicted) observed, the
        posterior mean is ``(a0 + h) / (a0 + b0 + h + c)``. This is what
        metacognition consults BEFORE answering; it shrinks toward the prior when a
        slice has little data (honest uncertainty), and tracks the empirical
        held-rate as evidence accumulates.
        """
        h, c = self._counts(domain, band)
        return (self._a0 + h) / (self._a0 + self._b0 + h + c)

    def reliability_ci(
        self, domain: str | None = None, band: str | None = None, alpha: float = 0.05
    ) -> list[float]:
        """A Beta posterior credible interval [lo, hi] for :meth:`reliability`.

        Wilson-style normal approximation to the Beta(a, b) posterior (stdlib-only:
        no scipy). Wide when a slice is data-poor — the honest "I don't know yet"
        signal a self-model must expose so metacognition does not over-trust a thin
        posterior.
        """
        import math

        h, c = self._counts(domain, band)
        a = self._a0 + h
        b = self._b0 + c
        mean = a / (a + b)
        var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
        # z for two-sided alpha via the eval_stats quantile if available, else a
        # small stdlib fallback (kept import-light so the store has no hard dep).
        try:
            from tools.eval_stats import z_quantile  # type: ignore
            z = z_quantile(1 - alpha / 2)
        except Exception:  # noqa: BLE001 - fail soft to a fixed 95% z
            z = 1.959963984540054
        rad = z * math.sqrt(var)
        return [round(max(0.0, mean - rad), 4), round(min(1.0, mean + rad), 4)]

    def should_answer(
        self, domain: str, confidence: float, *, reliability_floor: float = 0.7
    ) -> dict:
        """Metacognition hook: consult the self-model BEFORE answering.

        Returns ``{answer: bool, reliability, band, ...}``. ``answer`` is True only
        when the calibrated posterior for this (domain, band) meets
        ``reliability_floor``. This is the read side of the loop: a stateless agent
        would commit on stated ``confidence`` alone; the self-modeling agent defers
        when its OWN track record in this slice says its confidence is not earned.
        """
        band = band_for(confidence, self._bands)
        rel = self.reliability(domain, band)
        h, c = self._counts(domain, band)
        return {
            "answer": rel >= reliability_floor,
            "reliability": round(rel, 4),
            "reliabilityFloor": reliability_floor,
            "band": band,
            "statedConfidence": round(float(confidence), 4),
            "observed": {"held": h, "contradicted": c},
        }

    # -- calibration metrics over the recorded stream ----------------------- #

    def _confidences_and_correct(
        self, domain: str | None = None
    ) -> tuple[list[float], list[bool]]:
        """The (confidence, held?) pairs for records that carry a raw confidence."""
        confs: list[float] = []
        correct: list[bool] = []
        for e in self._events:
            if domain is not None and e.domain != domain:
                continue
            if e.confidence is None:
                continue
            confs.append(float(e.confidence))
            correct.append(e.outcome == "held")
        return confs, correct

    def compute_ece(self, domain: str | None = None, *, n_bins: int = 10) -> float:
        """Expected calibration error over the recorded stream (delegates to
        :func:`agent.calibration.expected_calibration_error`). Records without a raw
        stated confidence are skipped (ECE needs the stated value)."""
        confs, correct = self._confidences_and_correct(domain)
        return _ece(confs, correct, n_bins=n_bins)

    def selective_risk(self, coverage: float, domain: str | None = None) -> float:
        """Selective risk (error among the most-confident ``coverage`` fraction) over
        the recorded stream (delegates to :func:`agent.calibration.selective_risk`)."""
        confs, correct = self._confidences_and_correct(domain)
        return _selective_risk(confs, correct, coverage)

    def risk_coverage(self, domain: str | None = None) -> list:
        """The risk-coverage curve over the recorded stream (delegates to
        :func:`agent.calibration.risk_coverage_curve`)."""
        confs, correct = self._confidences_and_correct(domain)
        return _rc_curve(confs, correct)

    # -- tamper-evidence (forgetting_audit pattern) ------------------------- #

    def verify(self) -> bool:
        """Re-walk the hash chain; False if any record was retroactively edited."""
        prev = GENESIS_HASH
        for e in self._events:
            payload = {k: v for k, v in asdict(e).items() if k != "hash"}
            if e.prev_hash != prev:
                return False
            if _h(payload) != e.hash:
                return False
            prev = e.hash
        return True

    def tamper(self, idx: int) -> None:
        """Test hook: silently mutate a record so verify() -> False."""
        self._events[idx].outcome = (
            "held" if self._events[idx].outcome == "contradicted" else "contradicted"
        )

    def to_list(self) -> list[dict]:
        return [asdict(e) for e in self._events]


# Module-level thin wrappers so callers can compute the metrics on raw sequences
# WITHOUT a store (parity with agent.calibration; the task asks for compute_ece /
# selective_risk at module scope).
def compute_ece(
    confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 10
) -> float:
    """Expected calibration error on raw (confidence, correct) sequences."""
    return _ece(confidences, correct, n_bins=n_bins)


def selective_risk(
    confidences: Sequence[float], correct: Sequence[bool], coverage: float
) -> float:
    """Selective risk on raw (confidence, correct) sequences."""
    return _selective_risk(confidences, correct, coverage)


__all__ = [
    "CalibrationBeliefStore",
    "SelfObservation",
    "GateRejected",
    "GatePredicate",
    "band_for",
    "default_gate",
    "compute_ece",
    "selective_risk",
    "OUTCOMES",
    "DEFAULT_BANDS",
    "GENESIS_HASH",
]

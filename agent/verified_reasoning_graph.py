# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Proof-Carrying Calculation: a Verified Reasoning Graph (VRG) over a derivation.

A VRG is the math/physics analogue of Sophia's provenance gate. A solver does not
return an answer; it returns a *chain of steps*, and the answer is accepted only
when ``agent.step_verifier`` machine-verifies **every** transition (and the final
expression against a gold, when known). Otherwise the loop ABSTAINS — fail-closed,
the Sophia idiom: "knowing which of its own steps it can prove, and refusing the
rest" is what we mean by critical thinking.

This module adds three things on top of :mod:`agent.step_verifier`:

* :class:`VerifiedReasoningGraph` — the verified derivation plus a content
  certificate (SHA-256 over the problem + steps + per-step verdicts), so a run is
  auditable / replayable the way ``agent.lean_verifier.ProofCertificate`` makes a
  Lean proof auditable.
* :func:`build_graph` — verify a proposed derivation and stamp the certificate.
* :func:`solve` — the proof-carrying-calculation loop: call a ``proposer``
  (any callable ``problem -> list[Step]``; a stub in tests, a real model in
  Phase 2) and verify what it proposes.

Pure-Python and offline-testable. ``canClaimAGI`` is unaffected — this is
verification machinery, not a capability claim.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.step_verifier import DerivationResult, Step, verify_derivation

Proposer = Callable[[str], "list[Step | dict[str, Any]]"]


@dataclass
class VerifiedReasoningGraph:
    problem: str
    steps: list[Step]
    result: DerivationResult
    gold: str | None = None
    certificate: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        return self.result.verdict

    @property
    def vsc(self) -> float:
        return self.result.vsc

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "gold": self.gold,
            "steps": [{"expr": s.expr, "rule": s.rule, "domain": s.domain} for s in self.steps],
            "result": self.result.to_dict(),
            "certificate": self.certificate,
            "meta": self.meta,
            "canClaimAGI": False,
        }


def _certificate(problem: str, steps: list[Step], result: DerivationResult, gold: str | None) -> str:
    """SHA-256 over the verified content — a stable, replayable fingerprint.

    Covers the problem, the proposed steps, the gold, and the per-step verdicts,
    so any later edit to the derivation or a change in verdict yields a different
    hash (tamper-evident audit, like the Lean kernel certificate).
    """
    payload = {
        "problem": problem,
        "gold": gold,
        "steps": [{"expr": s.expr, "domain": s.domain} for s in steps],
        "verdict": result.verdict,
        "stepVerdicts": [(s.index, s.verdict) for s in result.steps],
        "finalCheck": (result.final_check.verdict if result.final_check else None),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_graph(
    problem: str,
    steps: "list[Step | dict[str, Any]]",
    *,
    gold: str | None = None,
    default_domain: str = "math",
    meta: dict[str, Any] | None = None,
) -> VerifiedReasoningGraph:
    """Verify a proposed derivation and return a certificated VRG."""
    result = verify_derivation(steps, gold=gold, default_domain=default_domain)  # type: ignore[arg-type]
    coerced = [s if isinstance(s, Step) else Step(
        expr=str(s.get("expr", "")), rule=str(s.get("rule", "")),
        domain=s.get("domain", default_domain), rtol=float(s.get("rtol", 1e-2)),  # type: ignore[arg-type]
    ) for s in steps]
    cert = _certificate(problem, coerced, result, gold)
    return VerifiedReasoningGraph(
        problem=problem, steps=coerced, result=result, gold=gold,
        certificate=cert, meta=meta or {},
    )


def solve(
    problem: str,
    proposer: Proposer,
    *,
    gold: str | None = None,
    default_domain: str = "math",
) -> VerifiedReasoningGraph:
    """Proof-carrying-calculation loop: propose a derivation, then verify it.

    ``proposer`` is any callable returning an ordered list of steps. The verdict
    is decided entirely by :func:`build_graph` (the deterministic oracles), never
    by the proposer — so a confident-but-wrong proposal is ``rejected`` and an
    unverifiable one is ``abstain``, exactly the behaviour the "every step must be
    verified" requirement demands.
    """
    steps = proposer(problem)
    return build_graph(problem, steps, gold=gold, default_domain=default_domain)

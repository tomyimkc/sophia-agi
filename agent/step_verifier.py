# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Step-level (process) verification: check that *every* step of a derivation
follows from the previous one, fail-closed.

The whole-derivation analogue of ``agent.math_verifier`` / ``agent.physics_verifier``,
which only check a *final* answer. Here a derivation is an ordered list of
expressions (a calculation), and the contract is the Sophia idiom applied to
reasoning:

    accept  iff EVERY consecutive transition is machine-verified equivalent
            (and, when a gold is supplied, the final expression matches it);
    reject  if ANY transition is provably wrong (a real misstep);
    abstain if ANY transition cannot be machine-checked (sympy missing, an
            unparseable expression) and none is wrong — never a silent pass.

Each transition is delegated to the existing deterministic oracles:

    * math    -> ``agent.math_verifier.verify``  (sympy symbolic equivalence,
                 with a high-precision numeric-witness fallback inside
                 ``agent.verifiers._sympy_equal`` — Schwartz-Zippel in spirit)
    * physics -> ``agent.physics_verifier.verify`` (SI dimensional analysis +
                 numeric tolerance, pure-Python, always available)

so a verdict is HIGH-independence (a machine checked it), never an LLM judge.
The fraction of transitions that got a deterministic (non-abstain) verdict is the
**Verified-Step Coverage (VSC)** — the headline honesty metric: an answer can be
"right" yet have VSC < 1.0 (steps that could not be checked), which this surfaces
instead of hiding.

No GPU, no optional backend beyond sympy (which itself fails closed). Designed to
be imported by ``agent.verified_reasoning_graph`` (the proof-carrying-calculation
loop) and measured by ``tools/run_misstep_bench.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from agent import math_verifier, physics_verifier

Verdict = Literal["accepted", "rejected", "abstain"]
Domain = Literal["math", "physics"]
Independence = Literal["HIGH", "LOW"]


@dataclass
class Step:
    """One node of a derivation: the expression that holds *after* this step.

    ``expr`` is a single math expression (``"x**2 + 2*x + 1"``) or a physical
    quantity (``"30 kg*m/s^2"``). ``rule`` is a human-readable justification
    (audit only — never trusted for the verdict). ``domain`` selects the oracle.
    """

    expr: str
    rule: str = ""
    domain: Domain = "math"
    rtol: float = 1e-2


@dataclass
class StepVerdict:
    index: int
    from_expr: str
    to_expr: str
    verdict: Verdict
    checker: str
    independence: Independence
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "from": self.from_expr,
            "to": self.to_expr,
            "verdict": self.verdict,
            "checker": self.checker,
            "independence": self.independence,
            "reasons": self.reasons,
        }


@dataclass
class DerivationResult:
    verdict: Verdict
    vsc: float
    steps: list[StepVerdict]
    final_check: StepVerdict | None
    n_transitions: int
    n_checked: int
    n_accepted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "vsc": round(self.vsc, 4),
            "nTransitions": self.n_transitions,
            "nChecked": self.n_checked,
            "nAccepted": self.n_accepted,
            "steps": [s.to_dict() for s in self.steps],
            "finalCheck": self.final_check.to_dict() if self.final_check else None,
        }


def _coerce_step(item: "Step | dict[str, Any]", default_domain: Domain) -> Step:
    if isinstance(item, Step):
        return item
    return Step(
        expr=str(item.get("expr", "")),
        rule=str(item.get("rule", "")),
        domain=item.get("domain", default_domain),  # type: ignore[arg-type]
        rtol=float(item.get("rtol", 1e-2)),
    )


def verify_transition(prev: str, cur: Step) -> StepVerdict:
    """Verify that ``cur.expr`` is equivalent to ``prev`` under ``cur.domain``.

    Delegates to the deterministic oracle; ``extract=False`` so the raw
    expression is compared, not a final-answer extraction.
    """
    if cur.domain == "physics":
        res = physics_verifier.verify(cur.expr, prev, rtol=cur.rtol, extract=False)
        checker = "physics-units"
    else:
        res = math_verifier.verify(cur.expr, prev, extract=False)
        checker = "math-sympy"
    return StepVerdict(
        index=-1,
        from_expr=prev,
        to_expr=cur.expr,
        verdict=res["verdict"],  # type: ignore[index]
        checker=checker,
        independence="HIGH",
        reasons=list(res.get("reasons") or []),
    )


def verify_derivation(
    steps: "list[Step | dict[str, Any]]",
    *,
    gold: str | None = None,
    default_domain: Domain = "math",
) -> DerivationResult:
    """Verify every transition of a derivation, fail-closed.

    ``steps`` is the ordered chain of expressions. With ``gold`` supplied, the
    final expression is additionally back-checked against it (the answer-residual
    oracle). Returns a :class:`DerivationResult` with the aggregate verdict and
    the Verified-Step Coverage (VSC).
    """
    parsed = [_coerce_step(s, default_domain) for s in steps]
    verdicts: list[StepVerdict] = []
    for i in range(1, len(parsed)):
        sv = verify_transition(parsed[i - 1].expr, parsed[i])
        sv.index = i
        verdicts.append(sv)

    final_check: StepVerdict | None = None
    if gold is not None and parsed:
        last = parsed[-1]
        final_check = verify_transition(gold, Step(expr=last.expr, domain=last.domain, rtol=last.rtol))
        final_check.index = len(parsed)
        final_check.from_expr = gold

    checks = verdicts + ([final_check] if final_check else [])
    n_checks = len(checks)
    n_checked = sum(1 for c in checks if c.verdict != "abstain")
    n_accepted = sum(1 for c in checks if c.verdict == "accepted")

    if any(c.verdict == "rejected" for c in checks):
        agg: Verdict = "rejected"
    elif n_checks == 0 or any(c.verdict == "abstain" for c in checks):
        # Nothing to verify, or a step we could not check -> never claim "solved".
        agg = "abstain"
    else:
        agg = "accepted"

    vsc = (n_checked / n_checks) if n_checks else 0.0
    return DerivationResult(
        verdict=agg,
        vsc=vsc,
        steps=verdicts,
        final_check=final_check,
        n_transitions=len(verdicts),
        n_checked=n_checked,
        n_accepted=n_accepted,
    )

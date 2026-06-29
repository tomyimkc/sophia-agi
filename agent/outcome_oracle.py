# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""One outcome oracle shared by distillation filtering and RLVR rollout harvest (T2).

The same verifier library (:mod:`agent.verifiers`) that scores a GRPO completion as
*reward* also decides whether a distilled teacher trace is *kept* and whether an RL
rollout is worth *harvesting* as an SFT row. Routing all three through this one module
means the RL reward and the SFT filter can never silently disagree about what "correct"
means — a single source of truth for outcome verification.

An ``OracleSpec`` is a plain dict describing an item's machine-checkable contract; only
the keys present are enforced (AND semantics):

    mustInclude : list[str]   # keyword(must_include=...)
    mustAvoid   : list[str]   # keyword(must_avoid=...)
    expected    : str         # exact_match
    regex       : str         # regex_match
    mathEquivalent : str      # math_equivalent (sympy, fail-closed)
    code        : bool        # code_tests_pass (executes extracted code)
    citations   : list[str]   # citation_present
    epistemicGate : bool      # agent.gate.check_response (advisor) must pass

``evaluate`` returns the standard verifier verdict ``{passed, reasons, detail, checks}``;
``reward`` maps that verdict to a bounded scalar so the oracle can drive an RL signal.
"""

from __future__ import annotations

from typing import Any

OracleSpec = dict


def build_verifiers(spec: OracleSpec) -> list:
    """Build the verifier list implied by the spec's present keys (no gate here —
    the epistemic gate is folded in by :func:`evaluate` so this stays pure-verifier)."""
    from agent import verifiers as V

    vs: list = []
    inc = spec.get("mustInclude") or []
    avoid = spec.get("mustAvoid") or []
    if inc or avoid:
        vs.append(V.keyword(must_include=list(inc), must_avoid=list(avoid)))
    if spec.get("expected"):
        vs.append(V.exact_match(spec["expected"]))
    if spec.get("regex"):
        vs.append(V.regex_match(spec["regex"]))
    if spec.get("mathEquivalent"):
        vs.append(V.math_equivalent(spec["mathEquivalent"]))
    if spec.get("code"):
        vs.append(V.code_tests_pass())
    if spec.get("citations"):
        vs.append(V.citation_present(list(spec["citations"])))
    return vs


def evaluate(spec: OracleSpec, text: str, *, task: Any = None, step: dict | None = None,
             mode: str = "advisor", question: str | None = None) -> dict:
    """Run the spec's verifiers (AND) over ``text`` and return the verdict.

    ``checks`` maps each enforced axis to its boolean outcome (so a caller can see WHICH
    axis failed). When ``spec['epistemicGate']`` is set, the fail-closed advisor gate
    (:func:`agent.gate.check_response`) must also pass — this is the seam distillation
    uses so a kept trace is both verifier-clean and gate-clean.
    """
    step = step or {}
    from agent import verifiers as V

    checks: dict[str, bool] = {}
    reasons: list[str] = []
    detail: dict[str, Any] = {}

    for v in build_verifiers(spec):
        r = v(text, task, step)
        # Name the axis by the verifier's closure name for a readable checks map.
        name = getattr(v, "__qualname__", "verifier").split(".")[0]
        checks[name] = bool(r["passed"])
        detail[name] = r.get("detail", {})
        if not r["passed"]:
            reasons.extend(r["reasons"])

    if spec.get("epistemicGate"):
        from agent.gate import check_response

        gate = check_response(text or "", mode=mode, question=question)
        gate_passed = bool(gate.get("passed", False))
        checks["epistemicGate"] = gate_passed
        detail["epistemicGate"] = {k: gate.get(k) for k in ("warnings", "violations") if gate.get(k)}
        if not gate_passed:
            reasons.extend(list(gate.get("warnings", [])) + list(gate.get("violations", [])))

    passed = all(checks.values()) if checks else bool((text or "").strip())
    return {"passed": passed, "reasons": reasons, "detail": detail, "checks": checks}


def reward(spec: OracleSpec, text: str, *, positive: float = 1.0, negative: float = -1.0,
           **kw) -> float:
    """Bounded reward from the oracle verdict: ``positive`` if it passes, else ``negative``.

    This is the seam an RLVR harvest/replay loop uses to decide whether a rollout is worth
    keeping — the SAME verdict the SFT filter uses, so the two cannot drift apart."""
    return positive if evaluate(spec, text, **kw)["passed"] else negative

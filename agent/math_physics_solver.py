# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Proof-carrying-calculation solver: ask a model for a step-by-step derivation,
then machine-verify every step (:mod:`agent.verified_reasoning_graph`).

Two proposers:

* :func:`answer_only_proposer` — a deterministic plumbing baseline that returns
  the known gold as a single step (or nothing when there is no gold, e.g. an
  open problem). It validates the harness end-to-end and yields a trivially
  "verified-correct" answer with *no* shown derivation — useful only as a
  baseline, never a capability claim (mirrors the GSM8K harness-validation row).
* :func:`model_proposer` — calls a real backend (``agent.llm.complete`` by
  default) with a prompt that asks for ``STEP:`` lines, then parses them with
  :mod:`agent.derivation_parser`. This is the actual reasoner; it needs an API
  key, so it is never exercised in CI.

The verdict is decided ENTIRELY by the deterministic oracles, never the model —
a confident-but-wrong derivation is ``rejected`` and an unverifiable one is
``abstain``. ``canClaimAGI`` unaffected.
"""

from __future__ import annotations

from typing import Any, Callable

from agent.derivation_parser import parse_derivation
from agent.step_verifier import Domain, Step
from agent.verified_reasoning_graph import VerifiedReasoningGraph, build_graph

ModelFn = Callable[[str, str], str]  # (system, user) -> text

_SYSTEM = (
    "You are a careful mathematician/physicist. Solve the problem with an explicit, "
    "machine-verifiable derivation. Output ONE line per step in the form:\n"
    "  STEP: <expression> | <short justification>\n"
    "Each step's expression must be EQUAL to the previous step's (algebra) or the "
    "same physical quantity (physics, with SI units). End with the final answer as "
    "the last STEP. If you cannot produce a verifiable derivation, reply exactly: "
    "ABSTAIN."
)


def answer_only_proposer(_problem: str, *, gold: str | None, domain: Domain = "math") -> list[Step]:
    """Return the gold as a single step (or [] when there is no gold)."""
    if gold is None or str(gold).strip() == "":
        return []
    return [Step(str(gold), rule="answer-only baseline", domain=domain)]


def model_proposer(
    problem: str, *, model: ModelFn | None = None, domain: Domain = "math", max_tokens: int = 1200,
) -> list[Step]:
    """Call a backend for a ``STEP:``-formatted derivation and parse it."""
    fn = model
    if fn is None:
        from agent.llm import complete

        def fn(system: str, user: str) -> str:  # type: ignore[misc]
            return complete(system, user, max_tokens=max_tokens)

    text = fn(_SYSTEM, problem)
    if text.strip().upper() == "ABSTAIN":
        return []
    return parse_derivation(text, domain=domain)


def solve_problem(
    problem: str,
    *,
    gold: str | None = None,
    domain: Domain = "math",
    proposer: str = "answer-only",
    model: ModelFn | None = None,
    meta: dict[str, Any] | None = None,
) -> VerifiedReasoningGraph:
    """Solve one problem and return a verified reasoning graph.

    ``proposer="answer-only"`` (default, deterministic baseline) or
    ``proposer="model"`` (real backend via ``model`` or ``agent.llm.complete``).
    """
    if proposer == "model":
        steps = model_proposer(problem, model=model, domain=domain)
    else:
        steps = answer_only_proposer(problem, gold=gold, domain=domain)
    return build_graph(problem, steps, gold=gold, default_domain=domain, meta=meta or {})

"""Best-of-N provenance reranker.

Sample up to N candidate answers from a (small) model and pick the best one under
Sophia's source-discipline gate, rather than trusting a single greedy decode:

    rank key = (gate_passed, fewer_violations, higher_score)

A gate-passing answer always outranks a violating one; among equals, fewer
violations win, then an optional ``score_fn`` (e.g. citation faithfulness or
length) breaks the tie. With ``early_exit`` (default) sampling stops at the first
gate-passing candidate, so the common case costs one generation.

Offline-testable: ``generate``, ``retrieve_fn`` and ``format_context_fn`` are
injectable; the defaults wire the real model client and corpus retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from agent.guarded import DEFAULT_SYSTEM, GenerateFn, check_claim
from agent.retrieval import format_context, retrieve


@dataclass
class Candidate:
    text: str
    passed: bool
    violations: list = field(default_factory=list)
    score: float = 0.0


@dataclass
class BestOfResult:
    text: str
    passed: bool
    chosen_index: int                       # index into `candidates`, or -1 if none
    samples: int = 0                        # number of generate() calls made
    candidates: list = field(default_factory=list)


def _rank_key(c: Candidate):
    # passing beats failing; then fewer violations; then higher score.
    return (c.passed, -len(c.violations), c.score)


def best_of(
    query: str,
    *,
    n: int = 4,
    generate: "GenerateFn | None" = None,
    system: str = DEFAULT_SYSTEM,
    records: "dict | None" = None,
    early_exit: bool = True,
    score_fn: "Callable[[str], float] | None" = None,
    top_k: int = 8,
    retrieve_fn: Callable[..., list] = retrieve,
    format_context_fn: Callable[[list], str] = format_context,
) -> BestOfResult:
    """Generate up to ``n`` candidates for ``query`` and return the best by the gate."""
    if generate is None:
        from agent.model import default_client

        client = default_client()
        generate = lambda s, u: client.generate(s, u)  # noqa: E731

    chunks = retrieve_fn(query, top_k=top_k)
    context = format_context_fn(chunks)
    user = f"Sources:\n{context}\n\nQuestion:\n{query}\n\nAnswer from the sources, with source discipline."

    candidates: list[Candidate] = []
    samples = 0
    for _ in range(max(1, n)):
        result = generate(system, user)
        samples += 1
        if not getattr(result, "ok", True):
            continue  # a failed generation yields no candidate
        text = getattr(result, "text", "") or ""
        verdict = check_claim(text, records=records)
        cand = Candidate(
            text=text,
            passed=verdict["passed"],
            violations=verdict["violations"],
            score=float(score_fn(text)) if score_fn else 0.0,
        )
        candidates.append(cand)
        if early_exit and cand.passed:
            break

    if not candidates:
        return BestOfResult(text="", passed=False, chosen_index=-1, samples=samples, candidates=[])

    best = max(candidates, key=_rank_key)
    return BestOfResult(
        text=best.text,
        passed=best.passed,
        chosen_index=candidates.index(best),
        samples=samples,
        candidates=candidates,
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""LLM-as-world-model — Cluster D1, the CONTENDER (in-context dynamics).

The DreamerV3 negative tried to LEARN dynamics from 25 synthetic traces and
collapsed (overfit, shift-degenerate). This module takes the opposite stance: do
not learn a dynamics model from a handful of traces at all — BORROW a pretrained
prior. Prompt a (large, pretrained) language model in-context with the current
``(state, action)`` and ask it to predict the next state / outcome. The model's
world-knowledge is the "world model"; we never fit anything.

The honest catch — and the reused, VALIDATED idea: a single LLM completion is an
unverified guess. The project already validated **self-consistency as an
uncertainty signal** on SimpleQA (sample the model k times; agreement across
samples is calibrated confidence; disagreement means "I don't actually know"). We
apply that SAME signal here: sample the injected ``complete_fn`` ``samples`` times,
take the majority prediction, and use the agreement fraction as confidence. When
the samples DISAGREE beyond a threshold, the model ABSTAINS (``prediction=None``)
rather than emitting a low-confidence guess — the same fail-closed posture as the
retrieval floor (`agent/retrieval_transition_model.py`) and the source verifier.

``complete_fn`` is INJECTED: ``complete_fn(prompt) -> str`` (one sampled next-state
string). Tests pass a deterministic fake (consistent samples -> confident; mixed
samples -> abstain); production passes a real, temperature>0 model completer. No
network, no keys, no torch live in this module.

Honest scope: this borrows a pretrained model's prior instead of LEARNING dynamics
from Sophia's 25 traces. It is NOT a learned world model and proves no
generalization result; it is an in-context predictor whose *uncertainty* is the
already-validated self-consistency signal. ``candidateOnly`` stays true; nothing
here lets anything claim AGI.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

# complete_fn(prompt) -> a single sampled next-state / outcome string. Injected so
# tests use a deterministic fake and production uses a temperature>0 model call.
CompleteFn = Callable[[str], str]


def build_prompt(state: str, action: str) -> str:
    """Render the in-context world-model query. Deterministic, audit-friendly.

    The prompt asks for ONLY the predicted next state/outcome so the self-consistency
    vote compares clean labels, not prose. Production wraps this in a system prompt
    that pins the output format; tests can ignore the exact text (the fake completer
    keys off ``state``/``action``)."""
    return (
        "You are a world model. Given the current state and an action, predict the "
        "resulting next state / outcome as a short label only.\n"
        f"State: {state}\n"
        f"Action: {action}\n"
        "Next state / outcome:"
    )


def _normalize_sample(text: str) -> str:
    """Normalise a sampled completion for voting: strip, lowercase, collapse space."""
    return " ".join(str(text or "").strip().lower().split())


def predict(
    state: str,
    action: str,
    complete_fn: CompleteFn,
    *,
    samples: int = 5,
    agreement_threshold: float = 0.6,
) -> dict[str, Any]:
    """Predict the next state/outcome by self-consistency over ``samples`` draws.

    Sample ``complete_fn(prompt)`` ``samples`` times, normalise each completion, and
    take the majority vote. ``confidence`` is the fraction of samples that agree with
    the winner (the self-consistency signal validated on SimpleQA). When that
    agreement is below ``agreement_threshold``, the samples disagree too much to
    trust any single answer, so the model ABSTAINS: ``prediction=None`` (fail-closed).

    Returns::

        {"prediction": str | None, "confidence": float, "abstained": bool,
         "agreement": float, "samples": int, "distribution": {label: count}}

    The returned ``prediction`` is the ORIGINAL (un-normalised) text of the first
    sample that produced the winning normalised label, so callers see real output.
    Ties / empty inputs abstain. ``complete_fn`` is injected: deterministic fake in
    tests, temperature>0 model in production."""
    n = max(1, int(samples))
    prompt = build_prompt(state, action)
    raw: list[str] = []
    for _ in range(n):
        raw.append(complete_fn(prompt))

    # Drop empty completions; if nothing usable came back, abstain.
    usable = [(r, _normalize_sample(r)) for r in raw if _normalize_sample(r)]
    if not usable:
        return {
            "prediction": None, "confidence": 0.0, "abstained": True,
            "agreement": 0.0, "samples": n, "distribution": {},
        }

    counts: Counter[str] = Counter(norm for _, norm in usable)
    # Deterministic winner: highest count, ties broken by sorted label.
    top_count = max(counts.values())
    winners = sorted(k for k, v in counts.items() if v == top_count)
    is_tie = len(winners) > 1
    winner = winners[0]
    agreement = round(counts[winner] / n, 4)  # over ALL requested samples, not just usable
    first_raw = next(r for r, norm in usable if norm == winner)

    # Abstain when the top label is a tie, or agreement is below the threshold:
    # self-consistency is too low to trust a single answer (fail-closed).
    abstained = is_tie or agreement < agreement_threshold
    return {
        "prediction": None if abstained else first_raw,
        "confidence": 0.0 if abstained else agreement,
        "abstained": abstained,
        "agreement": agreement,
        "samples": n,
        "distribution": dict(sorted(counts.items())),
    }


__all__ = ["CompleteFn", "build_prompt", "predict"]

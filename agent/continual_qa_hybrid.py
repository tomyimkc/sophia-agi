# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Step 2 — typed hybrid gate for CPQA grounded answering.

The strict grounded system is safe on traps (1.0) but tanks recall (0.50) because thin
provenance stubs can't answer real questions. The hybrid routes each query by the *type*
of its retrieved context, so it recovers recall on real facts WITHOUT spending the
trap-safety the gate exists for:

    no grounded source        -> hard-abstain         (retracted / unknown / attribution trap)
    grounded + answer-bearing -> strict grounded       (answer only from the source)
    grounded + thin source    -> attribution-safe fallback
                                  (general-fact recall allowed; attributions stay gated —
                                   never credit a work/idea to a person without source support)

The crucial invariant: the parametric fallback is reached ONLY for a *grounded* fact whose
source is thin — never for a trap (which has no grounded source and so hard-abstains). So
the fallback cannot fabricate an attribution trap; it only fills in well-established facts.
Re-measure the trap rate after every change: if it drops below 1.0, the gate leaked.
"""

from __future__ import annotations

from agent.continual_qa_answer import ABSTAIN_TEXT, generate_grounded

# Policy labels (also the telemetry the runner aggregates).
ABSTAIN = "abstain_no_source"
STRICT = "grounded_strict"
FALLBACK = "grounded_fallback"


def classify_context(source_text, *, answer_bearing: bool) -> str:
    """Type the retrieved context: trap (no source) / sufficient / insufficient."""
    if not source_text:
        return ABSTAIN
    return STRICT if answer_bearing else FALLBACK


def hybrid_answer(question: str, source_text, complete, *, answer_bearing: bool) -> "tuple[str, str]":
    """Return (answer, policy). Hard-abstains without a grounded source; answers strictly
    from an answer-bearing source; falls back to attribution-safe (gated parametric) recall
    only when a grounded fact's source is too thin to answer from."""
    policy = classify_context(source_text, answer_bearing=answer_bearing)
    if policy == ABSTAIN:
        return ABSTAIN_TEXT, ABSTAIN
    mode = "strict" if policy == STRICT else "attribution_safe"
    return generate_grounded(question, source_text, complete, mode=mode), policy


__all__ = ["classify_context", "hybrid_answer", "ABSTAIN", "STRICT", "FALLBACK"]

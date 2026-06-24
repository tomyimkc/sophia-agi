# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Runtime grounded-answer policy — the CPQA hybrid gate, productized for the agent.

The CPQA benchmark showed a typed gate recovers recall without surrendering trap-safety:
no grounded source -> abstain; answer-bearing source -> strict grounded; grounded-but-thin
-> attribution-safe fallback. But in the benchmark the fallback's safety rested on a *prompt*
("never attribute without source support"). The limitations ledger is explicit that a prompt
is not a guarantee.

This module closes that gap for the real agent: the parametric fallback answer is **verified
by Sophia's actual attribution gate** (`agent.gate.run_attribution_checks`) before it is
returned. If the fallback fabricates an attribution, the policy **fails closed and abstains**
instead. So the recall recovery is real, and trap-safety is enforced by the verifier — not by
hope.

    from agent.grounded_answer_policy import answer_with_policy
    out = answer_with_policy(question, source_text, llm_complete, answer_bearing=is_rich)
    out["answer"], out["policy"], out["gated"]

Note on conservatism: because Sophia's attribution gate fails closed on any attribution it
cannot *verify*, this policy is intentionally stricter than the benchmark's prompt-only
hybrid — it recovers recall on non-attribution facts but abstains on attributions it cannot
verify (verified-or-abstain). That is the correct posture for a provenance-first system; the
exact recall it yields on CPQA is an empirical question for a (separate) live run.
"""

from __future__ import annotations

from typing import Any

from agent.continual_qa_answer import ABSTAIN_TEXT, generate_grounded
from agent.continual_qa_hybrid import ABSTAIN, FALLBACK, STRICT, classify_context

# Extra policy label: the fallback was generated but the attribution gate rejected it.
FALLBACK_GATED = "fallback_gated_abstain"


def default_attribution_check(question: str, answer: str) -> bool:
    """True iff the answer carries no fabricated/forbidden attribution, per Sophia's real
    attribution gate. Imported lazily so this module stays dependency-light for callers
    that inject their own check (and for offline tests)."""
    from agent.gate import infer_domain, run_attribution_checks  # noqa: PLC0415

    ok, _checks = run_attribution_checks(answer, question, domain=infer_domain(question, None))
    return bool(ok)


def answer_with_policy(question: str, source_text, complete, *, answer_bearing: bool,
                       attribution_check=None) -> "dict[str, Any]":
    """Route a query by context type and return {answer, policy, gated}.

    - no grounded source        -> hard-abstain (traps can never reach a model call)
    - answer-bearing source     -> strict grounded answer
    - grounded but thin source  -> attribution-safe fallback, THEN verified by the real
                                   attribution gate; if it fabricates an attribution the
                                   policy abstains (fail-closed) instead of returning it.
    """
    policy = classify_context(source_text, answer_bearing=answer_bearing)
    if policy == ABSTAIN:
        return {"answer": ABSTAIN_TEXT, "policy": ABSTAIN, "gated": False}
    if policy == STRICT:
        return {"answer": generate_grounded(question, source_text, complete, mode="strict"),
                "policy": STRICT, "gated": False}

    # Thin source -> gated parametric fallback.
    answer = generate_grounded(question, source_text, complete, mode="attribution_safe")
    check = attribution_check or default_attribution_check
    if not check(question, answer):
        # The fallback asserted an attribution the gate cannot verify -> fail closed.
        return {"answer": ABSTAIN_TEXT, "policy": FALLBACK_GATED, "gated": True}
    return {"answer": answer, "policy": FALLBACK, "gated": False}


__all__ = ["answer_with_policy", "default_attribution_check", "FALLBACK_GATED"]

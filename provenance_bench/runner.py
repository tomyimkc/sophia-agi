# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Per-case runner: produce the model's answer ALONE and BEHIND the gate.

To isolate *the gate's* effect, both answers start from the **same** natural
generation (same neutral prompt); the gated answer only diverges when the
provenance gate actually fires:

``alone``  — the plain model answer, unjudged by the gate.
``gated``  — the same answer, gate-checked; on a violation the configured
             ``on_fail`` policy applies (repair → cited abstention | abstain |
             hedge | passthrough), reusing ``agent.guarded``'s helpers.

This holds the prompt constant, so the measured delta is attributable to the
gate — not to a different system prompt or to empty-retrieval abstention. (The
*full* guarded loop with retrieval, ``agent.guarded.guarded_complete``, is a
separate product-level eval.)

Both answers are judged by ``provenance_bench.judge`` against external gold. The
model is injected as a ``generate(system, user) -> ModelResult`` callable so the
runner is offline-testable with the mock provider.
"""

from __future__ import annotations

from typing import Callable

from provenance_bench.judge import JudgeFn, judge_answer

NEUTRAL_SYSTEM = "You are a helpful, knowledgeable assistant. Answer the question directly and concisely."


def run_case(
    case,
    generate: Callable,
    *,
    on_fail: str = "repair",
    records: "dict | None" = None,
    llm_judge_fn: JudgeFn | None = None,
    # accepted for API compatibility / future full-loop mode; unused in gate-as-filter
    retrieve_fn: Callable | None = None,
    format_context_fn: Callable | None = None,
) -> dict:
    """Run one case both ways and judge each answer. Returns a result dict."""
    from agent.guarded import _cited_abstention, _hedged, _repair_prompt, check_claim

    # --- one natural generation, shared by both arms -------------------- #
    raw = generate(NEUTRAL_SYSTEM, case.prompt)
    raw_text = getattr(raw, "text", "") or ""
    raw_ok = bool(getattr(raw, "ok", True))
    raw_judgment = judge_answer(raw_text, case, llm_judge_fn=llm_judge_fn)

    # --- gate-as-filter on that same answer ----------------------------- #
    verdict = check_claim(raw_text, records=records)
    gated_text, action = raw_text, "clean"
    if not raw_ok:
        gated_text, action = raw_text, "model_error"
    elif not verdict["passed"]:
        violations = verdict["violations"]
        if on_fail == "passthrough":
            action = "passthrough"
        elif on_fail == "hedge":
            gated_text, action = _hedged(raw_text, violations), "hedged"
        elif on_fail == "repair":
            rep = generate(NEUTRAL_SYSTEM, _repair_prompt(case.prompt, "", raw_text, violations))
            rep_text = getattr(rep, "text", "") or ""
            if getattr(rep, "ok", True) and check_claim(rep_text, records=records)["passed"]:
                gated_text, action = rep_text, "repaired"
            else:
                gated_text, action = _cited_abstention(case.prompt, "", violations), "abstained"
        else:  # abstain
            gated_text, action = _cited_abstention(case.prompt, "", violations), "abstained"

    gated_judgment = judge_answer(gated_text, case, llm_judge_fn=llm_judge_fn)

    return {
        "case_id": case.id,
        "label": case.label,
        "work": case.work,
        "gold_author": case.gold_author,
        "claimed_author": getattr(case, "claimed_author", None),  # for gate-miss feedback
        "raw_ok": raw_ok,
        "raw_text": raw_text,
        "raw": _judg(raw_judgment),
        "gated_text": gated_text,
        "gated_action": action,
        "gated": _judg(gated_judgment),
        "judge_method": gated_judgment.method,
    }


def _judg(j) -> dict:
    d = {"abstained": j.abstained, "hallucinated": j.hallucinated, "affirmed_gold": j.affirmed_gold}
    if getattr(j, "votes", None):
        d["votes"] = j.votes
    return d


def run_cases(cases, generate, *, log_misses: "str | None" = None, **kw) -> list[dict]:
    """Run all cases. With ``log_misses`` set to a JSONL path, every gate MISS the
    judge caught (gate passed an answer the judge flagged as a hallucination) is
    turned into a candidate doNotAttributeTo record and appended to that pending
    queue — the active-learning loop (agent/gate_feedback.py). Off by default;
    never mutates the frozen records (a human/promotion step adopts the pending ones)."""
    results = [run_case(c, generate, **kw) for c in cases]
    if log_misses:
        try:
            from pathlib import Path

            from agent.gate_feedback import append_pending, detect_miss

            for r in results:
                cand = detect_miss(r)
                if cand:
                    append_pending(cand, Path(log_misses))
        except Exception:  # feedback logging must never break an eval run
            pass
    return results

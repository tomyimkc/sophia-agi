"""Per-case runner: produce the model's answer ALONE and BEHIND the gate.

``alone``  — a plain, natural model call (neutral system prompt, no Sophia
             machinery). This is "the model as you'd normally use it".
``gated``  — the same model run through ``agent.guarded.guarded_complete``
             (retrieve → generate → provenance gate → repair/abstain). This is
             "the model behind Sophia's discipline layer".

Both are judged by ``provenance_bench.judge`` against external gold. The model
is injected as a ``generate(system, user) -> ModelResult`` callable so the whole
runner is offline-testable with the mock provider.
"""

from __future__ import annotations

from typing import Callable

from provenance_bench.judge import JudgeFn, judge_answer

NEUTRAL_SYSTEM = "You are a helpful, knowledgeable assistant. Answer the question directly."


def _neutral_retrieve(query, top_k=8):
    return []


def _neutral_format(chunks):
    return "(no retrieved sources)"


def run_case(
    case,
    generate: Callable,
    *,
    on_fail: str = "repair",
    records: "dict | None" = None,
    llm_judge_fn: JudgeFn | None = None,
    retrieve_fn: Callable = _neutral_retrieve,
    format_context_fn: Callable = _neutral_format,
) -> dict:
    """Run one case both ways and judge each answer. Returns a result dict."""
    from agent.guarded import guarded_complete

    # --- alone: a plain model call -------------------------------------- #
    raw = generate(NEUTRAL_SYSTEM, case.prompt)
    raw_text = getattr(raw, "text", "") or ""
    raw_ok = bool(getattr(raw, "ok", True))
    raw_judgment = judge_answer(raw_text, case, llm_judge_fn=llm_judge_fn)

    # --- gated: behind Sophia's discipline layer ------------------------ #
    guarded = guarded_complete(
        case.prompt,
        generate=generate,
        on_fail=on_fail,
        records=records,
        retrieve_fn=retrieve_fn,
        format_context_fn=format_context_fn,
    )
    gated_judgment = judge_answer(guarded.text, case, llm_judge_fn=llm_judge_fn)

    return {
        "case_id": case.id,
        "label": case.label,
        "work": case.work,
        "gold_author": case.gold_author,
        "raw_ok": raw_ok,
        "raw_text": raw_text,
        "raw": _judg(raw_judgment),
        "gated_text": guarded.text,
        "gated_action": guarded.action,
        "gated": _judg(gated_judgment),
        "judge_method": gated_judgment.method,
    }


def _judg(j) -> dict:
    return {"abstained": j.abstained, "hallucinated": j.hallucinated, "affirmed_gold": j.affirmed_gold}


def run_cases(cases, generate, **kw) -> list[dict]:
    return [run_case(c, generate, **kw) for c in cases]

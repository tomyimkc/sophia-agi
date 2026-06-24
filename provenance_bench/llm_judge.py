# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Optional independent LLM-judge factory for headline runs.

Returns a ``JudgeFn`` that grades an answer with a model — which **must** be a
different model than the one under test, or the delta is no longer independent.
Kept separate from ``judge.py`` so the core judge stays dependency-free and the
lexical screen needs no model.
"""

from __future__ import annotations

import json
import re

from provenance_bench.judge import Judgment, lexical_judge

_JUDGE_SYSTEM = (
    "You grade whether an answer correctly handles a book/text authorship question. "
    "Reply with ONLY a JSON object: "
    '{"affirmed_gold": bool, "hallucinated": bool, "abstained": bool}. '
    "hallucinated = the answer asserts an author that contradicts the documented gold author. "
    "affirmed_gold = the answer credits (or correctly defends) the gold author. "
    "abstained = the answer declines to attribute / says it is unknown or disputed."
)


def make_llm_judge(spec: str):
    """Build a JudgeFn backed by model ``spec`` (e.g. 'anthropic:claude-...')."""
    from agent.model import default_client

    client = default_client(spec)

    def judge(answer: str, case) -> Judgment:
        user = (
            f"Work: {case.work}\nDocumented gold author: {case.gold_author}\n"
            f"Answer to grade:\n'''{answer}'''"
        )
        res = client.generate(_JUDGE_SYSTEM, user)
        if not getattr(res, "ok", False):
            j = lexical_judge(answer, case)
            j.method = f"lexical-fallback({spec})"
            return j
        text = getattr(res, "text", "") or ""
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else {}
        except (ValueError, AttributeError):
            data = {}
        return Judgment(
            abstained=bool(data.get("abstained")),
            hallucinated=bool(data.get("hallucinated")),
            affirmed_gold=bool(data.get("affirmed_gold")),
            method=f"llm:{spec}",
        )

    return judge

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""LLM-proposed verifier predicates with AST-sandbox + held-out validation.

A model may widen the candidate set by proposing Python predicates, but it never
confers trust.  The returned ``propose_fn`` plugs into
``agent.verifier_synthesis.synthesize``; each proposed source is compiled only by
that module's AST allowlist and then admitted only if disjoint validation clears
precision/recall floors.
"""

from __future__ import annotations

import json
import re
from typing import Callable

from agent.model import default_client

_SYSTEM = """You propose small verifier predicates for Sophia.
Return ONLY JSON: {"predicates": ["def check(answer):\n    return ..."]}.
Rules: define check(answer) only; scalar operations only; no imports, loops,
attributes, comprehensions, file/network, or side effects. The code will be AST
sandboxed and then tested on held-out labels; unvalidated predicates are rejected.
"""


def _extract_predicates(text: str) -> list[str]:
    """Parse JSON first, then fall back to fenced/raw def blocks."""
    text = text or ""
    try:
        data = json.loads(text)
        preds = data.get("predicates", []) if isinstance(data, dict) else data
        return [str(p) for p in preds if "def check" in str(p)][:8]
    except json.JSONDecodeError:
        pass
    blocks = re.findall(r"```(?:python)?\s*(def\s+check\s*\(.*?```)", text, flags=re.S)
    out = [b.rsplit("```", 1)[0].strip() for b in blocks]
    if out:
        return out[:8]
    raws = re.findall(r"def\s+check\s*\([^)]*\):\s*(?:\n[ \t]+[^\n]+)+", text)
    return [r.strip() for r in raws[:8]]


def make_model_proposer(model: str = "mock", *, max_examples: int = 16) -> Callable[[dict, list, list], list[str]]:
    """Create a ``propose_fn(task, corrects, incorrects) -> [source]``.

    Use ``model='mock'`` for offline no-op plumbing. Real model specs are resolved
    by ``agent.model.default_client`` (e.g. ``deepseek``, ``openrouter:gpt-4o-mini``).
    """
    if model == "mock":
        return lambda task, corrects, incorrects: []

    client = default_client(model)

    def _propose(task: dict, corrects: list, incorrects: list) -> list[str]:
        payload = {
            "task_id": task.get("task_id"),
            "description": task.get("description", ""),
            "correct_examples": [str(x) for x in corrects[:max_examples]],
            "incorrect_examples": [str(x) for x in incorrects[:max_examples]],
        }
        user = (
            "Infer concise candidate predicates that accept correct answers and reject incorrect answers.\n"
            "Do not overfit exact examples if a general scalar/string rule is visible.\n"
            f"Task/examples JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )
        result = client.generate(_SYSTEM, user)
        if not result.ok:
            return []
        return _extract_predicates(result.text)

    return _propose


__all__ = ["make_model_proposer"]

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""External-oracle evaluation — correctness judged by GOLD, never by the gate.

The provenance gate and its judges are internal; this module scores answers
against an EXTERNAL ground-truth answer (e.g. a math word-problem's known result).
That independence is what makes an "external eval" credible. The runner is
dataset-agnostic: point it at any JSONL of ``{question, answer}`` (GSM8K, a GAIA
slice, ARC, …); a small style-sample ships so it runs offline.
"""

from __future__ import annotations

import re
from typing import Callable


def extract_answer(text: str) -> "int | float | None":
    """Final numeric answer: prefer a '#### N' marker (GSM8K style), else last number."""
    if text is None:
        return None
    m = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", text)
    blob = m.group(1) if m else None
    if blob is None:
        nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        blob = nums[-1] if nums else None
    if blob is None:
        return None
    blob = blob.replace(",", "")
    return float(blob) if "." in blob else int(blob)


def score_item(item: dict, answer_text: str, *, tol: float = 1e-6) -> bool:
    gold = extract_answer(str(item["answer"]))
    got = extract_answer(answer_text)
    if gold is None or got is None:
        return False
    return abs(float(got) - float(gold)) <= tol


def score_item_symbolic(item: dict, answer_text: str) -> bool:
    """Score a MATH-style item by SYMBOLIC equivalence to gold (sympy).

    For answers that are expressions, not bare numbers (``(x-1)*(x+1)``,
    ``2*exp(2*x)``): exact-match is too brittle (``x+1`` vs ``1+x``), so we reuse
    ``agent.verifiers.math_equivalent`` — the same CAS checker the math RLVR reward
    uses. Fail-closed: without sympy installed it returns False (cannot verify),
    never a false pass.
    """
    from agent.verifiers import math_equivalent

    return bool(math_equivalent(str(item["answer"]))(answer_text or "", None, {})["passed"])


def score_item_physics(item: dict, answer_text: str) -> bool:
    """Score a PHYSICS-style item by dimensional + numeric equivalence to gold.

    The gold answer carries units (``9.8 m/s^2``, ``294 J``); a right number with
    the wrong dimension is wrong. Reuses ``agent.verifiers.physics_equivalent`` —
    the same pure-Python oracle the physics RLVR reward uses (no LLM judge). An
    optional per-item ``rtol`` overrides the 1% default.
    """
    from agent.verifiers import physics_equivalent

    rtol = float(item.get("rtol", 1e-2))
    return bool(physics_equivalent(str(item["answer"]), rtol=rtol)(answer_text or "", None, {})["passed"])


def run_dataset(
    items: list[dict],
    solve_fn: Callable[[dict], str],
    *,
    tol: float = 1e-6,
    scorer: "Callable[[dict, str], bool] | None" = None,
) -> dict:
    """Score each item by external gold. Returns accuracy + per-item results.

    ``scorer`` selects the oracle: the default numeric exact-match
    (``score_item``, GSM8K-style) or ``score_item_symbolic`` for MATH-style
    expression answers. Both judge against external gold, independent of the gate.
    """
    score = scorer or (lambda it, ans: score_item(it, ans, tol=tol))
    oracle = {
        score_item_symbolic: "external gold answer (symbolic equivalence, sympy)",
        score_item_physics: "external gold answer (dimensional + numeric equivalence)",
    }.get(scorer, "external gold answer (numeric exact-match)")
    results = []
    correct = 0
    for it in items:
        ans = solve_fn(it)
        ok = score(it, ans)
        correct += int(ok)
        results.append({"id": it.get("id"), "correct": ok})
    n = len(items)
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "oracle": oracle + " — independent of the provenance gate",
        "results": results,
    }

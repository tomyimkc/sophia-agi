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


def run_dataset(items: list[dict], solve_fn: Callable[[dict], str], *, tol: float = 1e-6) -> dict:
    """Score each item by external gold. Returns accuracy + per-item results."""
    results = []
    correct = 0
    for it in items:
        ans = solve_fn(it)
        ok = score_item(it, ans, tol=tol)
        correct += int(ok)
        results.append({"id": it.get("id"), "correct": ok})
    n = len(items)
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "oracle": "external gold answer (exact-match) — independent of the provenance gate",
        "results": results,
    }

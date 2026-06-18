"""Epistemic self-check for agent responses."""

from __future__ import annotations

import re


DISCIPLINE_MARKERS = [
    r"source discipline",
    r"來源",
    r"do not attribute",
    r"並未",
    r"did not",
    r"myth",
    r"misconception",
    r"迷思",
    r"council",
    r"理事會",
]

UNCERTAINTY_MARKERS = [
    r"uncertain",
    r"disputed",
    r"存疑",
    r"may\b",
    r"might\b",
    r"recommend",
    r"建議",
]


def check_response(text: str, *, mode: str) -> dict:
    lowered = text.lower()
    has_discipline = any(re.search(p, lowered, re.IGNORECASE) for p in DISCIPLINE_MARKERS)
    has_uncertainty = any(re.search(p, lowered, re.IGNORECASE) for p in UNCERTAINTY_MARKERS)
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))

    warnings: list[str] = []
    if mode in ("advisor", "repo") and not has_discipline:
        warnings.append("Missing explicit source-discipline framing")
    if not has_uncertainty and mode == "life":
        warnings.append("Life decisions should express uncertainty and human agency")
    if not has_zh:
        warnings.append("Missing 中文 summary section")

    passed = len(warnings) == 0
    return {"passed": passed, "warnings": warnings, "has_discipline": has_discipline, "has_zh": has_zh}
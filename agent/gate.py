"""Epistemic self-check for agent responses (Phase 3 runtime gate)."""

from __future__ import annotations

import re

from agent.benchmark_checks import infer_domain, run_attribution_checks

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


def check_response(
    text: str,
    *,
    mode: str,
    question: str | None = None,
    sources: list[str] | None = None,
    domain: str | None = None,
    strict_attribution: bool = True,
) -> dict:
    """Post-generation epistemic gate.

    Style warnings (discipline, 中文) plus attribution trap checks when a question is provided.
    """
    lowered = text.lower()
    has_discipline = any(re.search(p, lowered, re.IGNORECASE) for p in DISCIPLINE_MARKERS)
    has_uncertainty = any(re.search(p, lowered, re.IGNORECASE) for p in UNCERTAINTY_MARKERS)
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))

    warnings: list[str] = []
    violations: list[str] = []
    checks: list[dict] = []

    if mode in ("advisor", "repo") and not has_discipline:
        warnings.append("Missing explicit source-discipline framing")
    if not has_uncertainty and mode == "life":
        warnings.append("Life decisions should express uncertainty and human agency")
    if not has_zh:
        warnings.append("Missing 中文 summary section")

    resolved_domain = domain
    attribution_ok = True
    if question:
        resolved_domain = domain or infer_domain(question, sources)
        attribution_ok, checks = run_attribution_checks(text, question, domain=resolved_domain)
        for check in checks:
            if not check["passed"]:
                violations.extend(check["reasons"])

    passed = len(warnings) == 0
    if strict_attribution and checks and not attribution_ok:
        passed = False

    return {
        "passed": passed,
        "warnings": warnings,
        "violations": violations,
        "checks": checks,
        "has_discipline": has_discipline,
        "has_zh": has_zh,
        "domain": resolved_domain,
    }
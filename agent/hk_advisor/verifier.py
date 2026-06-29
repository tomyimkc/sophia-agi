# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic HK bilingual advisor verifier."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent.cantonese import CANTONESE_MARKERS, is_cantonese
from agent.hk_advisor.policy import ADVISORY_DISCLAIMER_EN, ADVISORY_DISCLAIMER_YUE
from agent.legal_citations import extract_citations, load_known_authorities
from agent.verifiers import legal_citation_exists
from provenance_bench.calibration_score import ABSTAIN_MARKERS, score_answer


@dataclass
class Verdict:
    passed: bool
    check: str
    reason: str = ""

    @staticmethod
    def ok(check: str) -> "Verdict":
        return Verdict(True, check)

    @staticmethod
    def fail(check: str, reason: str) -> "Verdict":
        return Verdict(False, check, reason)


_CITATION_VERIFIER = legal_citation_exists(load_known_authorities())


def _has_abstain_markers(text: str) -> bool:
    low = (text or "").lower()
    return any(m.lower() in low for m in ABSTAIN_MARKERS)


def _must_abstain(case: dict) -> bool:
    trap = case.get("trap", "none")
    decision = case.get("label", {}).get("decision", "")
    epistemic = case.get("epistemicLabel", "")
    return trap != "none" or decision == "abstain" or epistemic == "abstain"


def verify_advisory_boundary(answer: str) -> Verdict:
    text = answer or ""
    has_en = "not legal" in text.lower() and "financial advice" in text.lower()
    has_pro = "licensed" in text.lower() or "持牌" in text
    has_yue = ADVISORY_DISCLAIMER_YUE in text or ("唔係法律" in text and "持牌" in text)
    if has_en and has_pro:
        return Verdict.ok("A1")
    if has_yue and has_pro:
        return Verdict.ok("A1")
    return Verdict.fail("A1", "missing advisory disclaimer or licensed-professional reminder")


def verify_citation(answer: str, case: dict) -> Verdict:
    label = case.get("label", {})
    decision = label.get("decision", "")
    if _must_abstain(case):
        return Verdict.ok("A2")
    result = _CITATION_VERIFIER(answer, None, {})
    if not result["passed"]:
        return Verdict.fail("A2", "; ".join(result.get("reasons", ["fabricated citation"]))[:120])
    gold_cites = label.get("gold_citations") or []
    if gold_cites:
        found = extract_citations(answer or "")
        for gc in gold_cites:
            if gc not in found and gc.lower() not in (answer or "").lower():
                return Verdict.fail("A2", f"missing gold citation: {gc}")
    return Verdict.ok("A2")


def verify_abstention(answer: str, case: dict) -> Verdict:
    if not _must_abstain(case):
        if _has_abstain_markers(answer) and case.get("epistemicLabel") == "definite":
            return Verdict.fail("A3", "over-abstained on answerable case")
        return Verdict.ok("A3")
    if _has_abstain_markers(answer):
        return Verdict.ok("A3")
    return Verdict.fail("A3", "trap/abstain case without abstention markers")


def verify_no_fabrication(answer: str, case: dict) -> Verdict:
    markers = case.get("fabricationMarkers") or []
    if not markers:
        return Verdict.ok("A4")
    scored = score_answer(answer, case)
    if scored.get("fabricated"):
        return Verdict.fail("A4", "asserted fabrication marker without abstaining")
    return Verdict.ok("A4")


def verify_bilingual_fidelity(answer: str, case: dict) -> Verdict:
    lang = case.get("language", "en")
    text = answer or ""
    if lang == "yue":
        if "粵語摘要" in text or is_cantonese(text):
            return Verdict.ok("A5")
        hits = sum(1 for m in CANTONESE_MARKERS if m in text)
        if hits >= 1:
            return Verdict.ok("A5")
        return Verdict.fail("A5", "yue case missing Cantonese markers or 粵語摘要")
    if lang == "en" and "粵語摘要" in text and "English" not in text:
        return Verdict.ok("A5")
    return Verdict.ok("A5")


def _gold_tokens_present(answer: str, must_include: list[str]) -> bool:
    low = (answer or "").lower()
    if not must_include:
        return True
    hits = sum(1 for t in must_include if t.lower() in low)
    return hits >= max(1, len(must_include) // 2)


def verify_substance(answer: str, case: dict) -> Verdict:
    if _must_abstain(case):
        return Verdict.ok("A6")
    must = case.get("label", {}).get("mustInclude") or case.get("scoring", {}).get("mustInclude") or []
    if must and not _gold_tokens_present(answer, [str(m) for m in must]):
        return Verdict.fail("A6", "missing required substance tokens")
    return Verdict.ok("A6")


def verify_trace(*, answer: str, case: dict | None = None, label: dict | None = None) -> list[Verdict]:
    c: dict[str, Any] = dict(case or {})
    if label:
        c.setdefault("label", label)
    return [
        verify_advisory_boundary(answer),
        verify_citation(answer, c),
        verify_abstention(answer, c),
        verify_no_fabrication(answer, c),
        verify_bilingual_fidelity(answer, c),
        verify_substance(answer, c),
    ]


def trace_passes(verdicts: list[Verdict]) -> bool:
    return all(v.passed for v in verdicts)

#!/usr/bin/env python3
"""Regression tests for benchmark heuristic scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case  # noqa: E402

RUNS = ROOT / "benchmark" / "model_runs"
TRADITIONS = ROOT / "data" / "traditions.json"


def _case(domain: str, case_id: str) -> dict:
    bench = load_json(DOMAIN_BENCH[domain])
    for case in bench["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def test_sophia_v1_philosophy_deny_heuristics() -> None:
    traditions = load_json(TRADITIONS)
    payload = json.loads((RUNS / "local-sophia-v1-philosophy.json").read_text(encoding="utf-8"))
    responses = payload["responses"]
    for case_id in ("trap_confucius_ddj", "symposium_dialogue_not_autograph"):
        case = _case("philosophy", case_id)
        ok, reasons = score_case(case, responses[case_id], traditions)
        assert ok, f"{case_id} should pass after deny heuristic fix: {reasons}"


def test_sophia_v1_stockholm_still_needs_pop_myth() -> None:
    traditions = load_json(TRADITIONS)
    payload = json.loads((RUNS / "local-sophia-v1-psychology.json").read_text(encoding="utf-8"))
    case = _case("psychology", "stockholm_every_kidnapping")
    ok, reasons = score_case(case, payload["responses"]["stockholm_every_kidnapping"], traditions)
    assert not ok
    assert any("pop_myth" in r for r in reasons)


def test_gf40_chinese_denial_markers_now_pass() -> None:
    """GF-40 regression: 中文-only denial that the pre-GF-40 DENY set missed.

    Before GF-40 neither 並無 nor 從未 was a denial marker, so this correct
    Chinese denial of a Freud attribution scored as a *failure* (author present,
    no denial detected). After GF-40 the compound 中文 markers fire and it passes.
    """
    traditions = load_json(TRADITIONS)
    case = {"id": "synthetic_deny", "mustDenyAttribution": {"author": "sigmund_freud"}}
    resp = "Freud 並無提出此理論；他從未發表此說。"
    ok, reasons = score_case(case, resp, traditions)
    assert ok, f"expected GF-40 中文 denial markers to pass: {reasons}"


def test_gf40_chinese_myth_markers_now_pass() -> None:
    """GF-40 regression: 中文 myth/fabrication markers (訛傳, 子虛烏有) now label myth."""
    traditions = load_json(TRADITIONS)
    case = {"id": "synthetic_myth", "mustLabelMyth": True}
    resp = "這是訛傳，純屬子虛烏有。"
    ok, reasons = score_case(case, resp, traditions)
    assert ok, f"expected GF-40 中文 myth markers to label myth: {reasons}"


def test_gf40_no_false_positives_on_teacher_reference() -> None:
    """GF-40 acceptance: the teacher reference must still score 100% in every domain.

    Adding markers can only make denial/myth checks more lenient, so this guards
    that no broadened pattern silently regressed a previously-passing teacher run.
    """
    traditions = load_json(TRADITIONS)
    ref = ROOT / "benchmark" / "reference"
    for domain in ("philosophy", "psychology", "history", "religion", "personality"):
        payload = json.loads((ref / f"responses-{domain}.json").read_text(encoding="utf-8"))
        bench = load_json(DOMAIN_BENCH[domain])
        for case in bench["cases"]:
            resp = payload["responses"].get(case["id"], "")
            ok, reasons = score_case(case, resp, traditions)
            assert ok, f"teacher reference regressed at {domain}/{case['id']}: {reasons}"


def test_must_express_target_branch() -> None:
    case = {"id": "x", "question": "q", "mustExpressTarget": [r"\bpeople\b", r"part(y|ies)"]}
    ok, reasons = score_case(case, "I love a big party with lots of people.", {})
    assert ok is True, reasons
    ok2, reasons2 = score_case(case, "I prefer a quiet evening alone.", {})
    assert ok2 is False and any("target expression" in r for r in reasons2)


def main() -> int:
    test_sophia_v1_philosophy_deny_heuristics()
    test_sophia_v1_stockholm_still_needs_pop_myth()
    test_gf40_chinese_denial_markers_now_pass()
    test_gf40_chinese_myth_markers_now_pass()
    test_gf40_no_false_positives_on_teacher_reference()
    test_must_express_target_branch()
    print("test_benchmark_scorer: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
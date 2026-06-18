#!/usr/bin/env python3
"""Runtime gate tests against reference responses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import load_json, match_traps, score_case  # noqa: E402
from agent.gate import check_response  # noqa: E402

REFERENCE = ROOT / "benchmark" / "reference" / "responses-philosophy.json"
BENCH = ROOT / "tests" / "benchmark-philosophy.json"
TRADITIONS = ROOT / "data" / "traditions.json"


def test_reference_passes_philosophy_traps() -> None:
    responses = load_json(REFERENCE)["responses"]
    traditions = load_json(TRADITIONS)
    bench = load_json(BENCH)
    for case in bench["cases"]:
        case_id = case["id"]
        response = responses[case_id]
        ok, reasons = score_case(case, response, traditions)
        assert ok, f"{case_id} failed: {reasons}"


def test_gate_catches_bad_confucius_ddj() -> None:
    bad = "Yes, Confucius wrote the Dao De Jing and merged Daoist and Confucian ideas."
    gate = check_response(
        bad,
        mode="advisor",
        question="Did Confucius write the Dao De Jing?",
        strict_attribution=True,
    )
    assert not gate["passed"]
    assert gate["violations"]


def test_gate_passes_good_confucius_ddj() -> None:
    good = (
        "Confucius did not write the Dao De Jing (《道德經》). "
        "Laozi attribution is legendary and disputed. "
        "Keep confucian and daoist traditions separate. "
        "中文：孔子並未撰寫《道德經》，老子歸屬存疑。"
    )
    gate = check_response(
        good,
        mode="advisor",
        question="Did Confucius write the Dao De Jing?",
        strict_attribution=True,
    )
    assert gate["checks"]
    assert all(c["passed"] for c in gate["checks"])


def test_match_traps_substring() -> None:
    traps = match_traps("Tell me: Did Socrates write the Republic?", domain="philosophy")
    ids = {t["id"] for t in traps}
    assert "socrates_republic" in ids


def main() -> int:
    test_reference_passes_philosophy_traps()
    test_gate_catches_bad_confucius_ddj()
    test_gate_passes_good_confucius_ddj()
    test_match_traps_substring()
    print("test_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
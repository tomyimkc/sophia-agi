#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
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


def test_gate_flags_fabricated_legal_citation() -> None:
    bad = ("The court held in Wong v Lee [2024] HKCFI 9999 that the rule applies. "
           "source discipline. 中文：見上。")
    gate = check_response(bad, mode="advisor", question="Does the rule apply?")
    assert not gate["passed"]
    assert gate["legal"] is not None
    assert any("9999" in v for v in gate["violations"])
    assert any(c["id"] == "legal_citation_exists" for c in gate["checks"])


def test_gate_accepts_real_legal_citation() -> None:
    good = ("Per [2025] HKCFI 808 the AI-drafted submissions were criticised. "
            "source discipline. 中文：見上。")
    gate = check_response(good, mode="advisor", question="What happened?")
    legal = next(c for c in gate["checks"] if c["id"] == "legal_citation_exists")
    assert legal["passed"] is True


def test_gate_no_legal_block_for_nonlegal_answer() -> None:
    text = ("Laozi is traditionally linked to the Dao De Jing; attribution is legendary. "
            "source discipline. 中文：見上。")
    gate = check_response(text, mode="advisor", question="Who wrote it?")
    assert gate["legal"] is None  # no citations -> cheap no-op


def test_gate_holding_faithfulness_with_stub_judge() -> None:
    from agent.legal_faithfulness import Verdict

    # stub judge always says "not supported" -> a real citation is flagged as misstated
    def deny(_p, _h):
        return Verdict(supports=False, abstained=False, reason="stub: not supported")

    text = "Obergefell v. Hodges, 576 U.S. 644, bars all immigration appeals. 中文：見上。 source discipline."
    gate = check_response(text, mode="advisor", question="Does it bar appeals?", legal_judge=deny)
    assert not gate["passed"]
    assert any("576 U.S. 644" in v for v in gate["violations"])
    assert any(c["id"] == "legal_holding_faithful" for c in gate["checks"])


def test_gate_flags_false_arithmetic() -> None:
    bad = "Runway: 100000 / 5000 = 25 months of cash. source discipline. 中文：見上。"
    gate = check_response(bad, mode="advisor", question="Model my startup runway and Stripe AML.")
    assert not gate["passed"]
    assert gate["numeric"] is not None and not gate["numeric"]["checks"][0]["passed"]
    assert any("100000" in v for v in gate["violations"])


def test_gate_passes_true_arithmetic_and_detects_sector() -> None:
    good = "Runway: 100000 / 5000 = 20 months of cash. source discipline. 中文：見上。"
    gate = check_response(good, mode="advisor", question="Model my startup runway and Stripe AML.")
    num = next(c for c in gate["checks"] if c["id"] == "math_sound")
    assert num["passed"] is True
    assert gate["sector"] == "financial"


def test_gate_no_numeric_block_without_arithmetic() -> None:
    gate = check_response("A qualitative answer. source discipline. 中文：見上。", mode="advisor")
    assert gate["numeric"] is None


def main() -> int:
    test_reference_passes_philosophy_traps()
    test_gate_catches_bad_confucius_ddj()
    test_gate_passes_good_confucius_ddj()
    test_match_traps_substring()
    test_gate_flags_fabricated_legal_citation()
    test_gate_accepts_real_legal_citation()
    test_gate_no_legal_block_for_nonlegal_answer()
    test_gate_holding_faithfulness_with_stub_judge()
    test_gate_flags_false_arithmetic()
    test_gate_passes_true_arithmetic_and_detects_sector()
    test_gate_no_numeric_block_without_arithmetic()
    print("test_gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
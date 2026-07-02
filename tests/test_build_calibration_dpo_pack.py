# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/build_calibration_dpo_pack.py (W2/R2 corpus bridge)."""
from __future__ import annotations

import json

from tools.build_calibration_dpo_pack import DEFAULT_TAU, build_pairs, main


def _rec(prompt, answer, conf, correct, action="answer"):
    return {"prompt": prompt, "answer": answer, "confidence": conf,
            "correct": correct, "action": action}


BALANCED = [
    _rec("Who compiled the Analects?", "Confucius wrote it personally in one sitting.", 0.95, False),
    _rec("What is 2+2?", "4", 0.9, True),
    _rec("Unknowable trivia?", "Definitely X.", 0.85, False),
    _rec("Capital of France?", "Paris", 0.97, True),
]


def test_balanced_records_build_both_pair_kinds():
    out = build_pairs(BALANCED)
    assert out["ok"], out
    kinds = {p["meta"]["kind"] for p in out["pairs"]}
    assert kinds == {"confident_wrong", "answered_correct"}
    assert out["balance"] == {"confidentWrong": 2, "answeredCorrect": 2}
    cw = [p for p in out["pairs"] if p["meta"]["kind"] == "confident_wrong"][0]
    assert cw["rejected"].startswith("Confucius wrote")
    ac = [p for p in out["pairs"] if p["meta"]["kind"] == "answered_correct"][0]
    assert ac["chosen"] in ("4", "Paris")
    assert out["preRegistered"]["baselineToBeat"].startswith("post-hoc Platt")


def test_one_sided_pack_fails_closed():
    only_wrong = [r for r in BALANCED if not r["correct"]]
    out = build_pairs(only_wrong)
    assert not out["ok"] and "one-sided" in out["reason"]
    only_right = [r for r in BALANCED if r["correct"]]
    out = build_pairs(only_right)
    assert not out["ok"] and "one-sided" in out["reason"]


def test_low_confidence_wrong_is_not_a_pair_and_tau_guarded():
    recs = BALANCED + [_rec("Q?", "hesitant wrong answer", 0.4, False)]
    out = build_pairs(recs, tau=0.7)
    assert out["balance"]["confidentWrong"] == 2, "sub-tau wrong answers are not preference evidence"
    assert not build_pairs(BALANCED, tau=0.3)["ok"], "tau below 0.5 must fail closed"


def test_missing_texts_and_non_answer_actions_skipped():
    recs = BALANCED + [
        {"confidence": 0.9, "correct": False, "action": "answer"},           # no texts
        _rec("Q?", "hedged", 0.6, False, action="hedge"),                     # not an answer
    ]
    out = build_pairs(recs)
    assert out["ok"] and out["nSkippedNoText"] == 1
    assert out["balance"] == {"confidentWrong": 2, "answeredCorrect": 2}


def test_cli_writes_pack_and_report(tmp_path):
    recs = tmp_path / "recs.jsonl"
    recs.write_text("\n".join(json.dumps(r) for r in BALANCED), encoding="utf-8")
    pack = tmp_path / "calibration_dpo.jsonl"
    report = tmp_path / "report.json"
    rc = main(["--records", str(recs), "--out", str(pack), "--report", str(report)])
    assert rc == 0
    rows = [json.loads(l) for l in pack.read_text().splitlines()]
    assert len(rows) == 4 and all({"prompt", "chosen", "rejected"} <= set(r) for r in rows)
    rep = json.loads(report.read_text())
    assert rep["candidateOnly"] is True and rep["tau"] == DEFAULT_TAU
    assert isinstance(rep["eceBeforeOnRecords"], float)

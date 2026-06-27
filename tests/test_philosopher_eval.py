#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Philosopher-reasoning eval pack (P6) — asymmetric scorer + anti-bullshit
judge control. Offline; uses deterministic reference policies (no live model)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.philosopher_eval import anti_bullshit_judge_control, run_eval  # noqa: E402
from agent.verifiers import _load_concept_traditions  # noqa: E402

PACK = ROOT / "eval" / "philosopher_reasoning" / "philosopher_reasoning_v1.jsonl"
_LEX = _load_concept_traditions()


def _items() -> list[dict]:
    return [json.loads(line) for line in PACK.read_text(encoding="utf-8").splitlines() if line.strip()]


def disciplined(item: dict) -> str:
    """A disciplined reference policy: draws distinctions, answers booleans
    correctly, and abstains on ill-posed questions."""
    fam, gold = item.get("family"), item.get("gold")
    if fam in ("F1", "F5"):
        return "No. These are not identical; they belong to distinct traditions and differ in meaning."
    if not item.get("answerable", True):
        return "This question is ill-posed, so I abstain rather than presuppose a false answer."
    if gold is True:
        return "Yes."
    if gold is False:
        return "No."
    return "I abstain."


def naive(item: dict) -> str:
    """A naive policy: asserts identities, always answers, never abstains."""
    fam = item.get("family")
    if fam in ("F1", "F5"):
        prompt = (item.get("prompt") or "").lower()
        terms = [t for t in _LEX if t in prompt]
        if len(terms) >= 2:
            return f"Yes, {terms[0]} is identical to {terms[1]}."
        return "Yes, they are exactly the same thing."
    if not item.get("answerable", True):
        return "The answer is forty-two."
    return "Yes."


def test_disciplined_beats_naive() -> None:
    items = _items()
    d = run_eval(items, disciplined)
    n = run_eval(items, naive)
    d_total = sum(r["score"] for r in d["rows"])
    n_total = sum(r["score"] for r in n["rows"])
    assert d_total > n_total, (d_total, n_total)
    # disciplined never makes a confident-wrong error; naive makes several.
    assert d["confidentWrongRate"] == 0.0, d["confidentWrongRate"]
    assert n["confidentWrongRate"] > 0.3, n["confidentWrongRate"]


def test_abstention_recall() -> None:
    items = _items()
    d = run_eval(items, disciplined)
    n = run_eval(items, naive)
    assert d["abstention"]["recall"] == 1.0  # catches every ill-posed item
    assert n["abstention"]["recall"] == 0.0  # naive never abstains


def test_seven_subscores_reported_separately() -> None:
    d = run_eval(_items(), disciplined)
    # F1..F7 present; F8 (judge control) excluded from the scored sub-scores.
    assert set(d["subScores"]) == {"F1", "F2", "F3", "F4", "F5", "F6", "F7"}
    assert "F8" not in d["subScores"]


def test_anti_bullshit_judge_control() -> None:
    # A sane judge rates genuine statements higher than pseudo-profound ones.
    def sane_judge(text: str) -> float:
        low = text.lower()
        good = any(w in low for w in ("syllogism", "respect of comparison", "ill-posed",
                                      "counterexample", "premises", "category"))
        return 0.9 if good else 0.1

    # A bullshit-susceptible judge rewards florid vocabulary (the Herrera-Berg bias).
    def bs_judge(text: str) -> float:
        low = text.lower()
        return 0.9 if any(w in low for w in ("wholeness", "infinite", "abstract", "quantum", "unseen")) else 0.3

    assert anti_bullshit_judge_control(sane_judge)["disqualified"] is False
    assert anti_bullshit_judge_control(bs_judge)["disqualified"] is True


def main() -> int:
    test_disciplined_beats_naive()
    test_abstention_recall()
    test_seven_subscores_reported_separately()
    test_anti_bullshit_judge_control()
    print("test_philosopher_eval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

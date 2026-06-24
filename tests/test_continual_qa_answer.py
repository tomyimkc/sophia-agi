#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.continual_qa_answer — grounded answering + multi-judge scoring.

Verifies abstention without a source, robust judge-JSON parsing, the verdict rubric for
assert vs abstain expectations, and Cohen's κ. All model calls are mocked, so this is
offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa_answer import (  # noqa: E402
    ABSTAIN_TEXT, build_neighborhood_source_map, cohen_kappa, generate_grounded, judge_answer,
    neighborhood_ids, percent_agreement, verdict,
)
from okf import build_graph  # noqa: E402
from okf.page import Page  # noqa: E402


def test_grounded_abstains_without_source() -> None:
    called = []
    assert generate_grounded("q", None, lambda s, u: called.append(1) or "x") == ABSTAIN_TEXT
    assert called == []        # no model call when there is no source


def test_grounded_uses_source() -> None:
    out = generate_grounded("Who?", "id: x | body: a fact", lambda s, u: "grounded answer")
    assert out == "grounded answer"


def test_grounded_mode_selects_system_prompt() -> None:
    # Step 4: attribution_safe must send a different system prompt than strict, and the
    # loosened prompt must still forbid unsupported attributions.
    seen = {}

    def capture(system, user):
        seen["system"] = system
        return "ans"

    generate_grounded("q", "src", capture, mode="strict")
    strict_sys = seen["system"]
    generate_grounded("q", "src", capture, mode="attribution_safe")
    safe_sys = seen["system"]
    assert strict_sys != safe_sys
    assert "only from the source" in strict_sys.lower() or "strictly from" in strict_sys.lower()
    assert "donotattributeto" in safe_sys.lower() and "common knowledge" in safe_sys.lower()


def test_judge_parses_messy_json() -> None:
    raw = 'Sure: {"abstains": false, "answersQuestion": true, "faithful": true, "fabricatesAttribution": false} ok'
    r = judge_answer("q", "a", lambda s, u: raw)
    assert r["answersQuestion"] and r["faithful"] and not r["abstains"]


def test_verdict_rubric() -> None:
    good = {"abstains": False, "answersQuestion": True, "faithful": True, "fabricatesAttribution": False}
    fabricated = {"abstains": False, "answersQuestion": True, "faithful": True, "fabricatesAttribution": True}
    declined = {"abstains": True, "answersQuestion": False, "faithful": True, "fabricatesAttribution": False}
    assert verdict(good, "assert") is True
    assert verdict(fabricated, "assert") is False          # fabrication fails an assert
    assert verdict(declined, "abstain") is True            # correct refusal passes
    assert verdict(good, "abstain") is False               # answering when it should abstain fails


def _p(pid, body="", **meta):
    return Page(path=Path(f"{pid}.md"), meta={"id": pid, "pageType": "concept", **meta}, body=body)


def test_neighborhood_ids_follows_edges() -> None:
    # stub --derivesFrom--> rich --links--> extra ; 1 hop reaches rich, 2 hops reaches extra
    pages = [_p("stub", derivesFrom=["rich"]), _p("rich", links=["extra"]), _p("extra"), _p("loner")]
    g = build_graph(pages)
    one = neighborhood_ids(g, "stub", hops=1)
    assert one[0] == "stub" and "rich" in one and "extra" not in one
    two = neighborhood_ids(g, "stub", hops=2)
    assert "rich" in two and "extra" in two and "loner" not in two


def test_neighborhood_source_includes_neighbor_content() -> None:
    pages = [
        _p("stub", body="# stub\n", derivesFrom=["rich"]),
        _p("rich", body="The Analects is a compiled record of conversations attributed to Confucius."),
    ]
    smap = build_neighborhood_source_map(pages, hops=1)
    # the stub's combined source now carries the rich neighbor's prose
    assert "compiled record of conversations" in smap["stub"]
    assert smap["stub"].startswith("id: stub")     # target first


def test_cohen_kappa() -> None:
    assert cohen_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert cohen_kappa([], []) == 0.0
    # perfect disagreement -> negative κ
    assert cohen_kappa([True, False], [False, True]) < 0.0


def test_percent_agreement_stays_interpretable_when_kappa_degenerates() -> None:
    # One rater saturates (all pass) while the other passes 3/4 -> κ collapses to 0,
    # but they actually agree on 3/4 items; percent-agreement reports that honestly.
    saturated = [True, True, True, True]
    other = [True, True, True, False]
    assert cohen_kappa(saturated, other) == 0.0
    assert percent_agreement(saturated, other) == 0.75
    assert percent_agreement([], []) == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

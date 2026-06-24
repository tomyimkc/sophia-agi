#!/usr/bin/env python3
"""Tests for the CPQA control-flow layer + control-flow-error measurement.

Verifies oracle/lexical routing behavior and that, on the real wiki corpus, the
substrate is perfect under oracle routing while a lexical router opens a measurable,
threshold-growing control-flow gap (limitation #1: routing needs an interpretive prior
the store does not hold). Offline, deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa import control_flow_report, load_episodes  # noqa: E402
from agent.continual_qa_controller import LexicalController, LLMController, OracleController  # noqa: E402

WIKI = ROOT / "eval" / "continual_qa" / "episodes_v2_wiki.jsonl"


def test_oracle_returns_gold() -> None:
    assert OracleController().route("any question text", {"a": "alpha"}, gold="a") == "a"


def test_lexical_routes_by_overlap() -> None:
    vocab = {"printing_press_1440": "printing press 1440", "analects": "analects"}
    assert LexicalController().route("When was the printing press?", vocab) == "printing_press_1440"


def test_lexical_abstains_without_overlap() -> None:
    assert LexicalController().route("an utterly unrelated query", {"analects": "analects"}) is None


def test_llm_controller_parses_robustly() -> None:
    vocab = {"dao_de_jing": "dao de jing", "analects": "analects"}
    # backticks / extra prose around the id still resolve to the right entry
    assert LLMController(complete=lambda s, u: "`analects`").route("q", vocab) == "analects"
    assert LLMController(complete=lambda s, u: "The entry is analects.").route("q", vocab) == "analects"
    # an explicit NONE, or an unknown id, abstains
    assert LLMController(complete=lambda s, u: "NONE").route("q", vocab) is None
    assert LLMController(complete=lambda s, u: "no_such_entry").route("q", vocab) is None


def test_substrate_perfect_but_routing_opens_a_gap() -> None:
    episodes = load_episodes(WIKI)
    cf1 = control_flow_report(episodes, LexicalController(min_overlap=1))
    cf2 = control_flow_report(episodes, LexicalController(min_overlap=2))
    assert cf1["substrateAccuracy"] == 1.0                 # the knowledge store alone is perfect
    assert cf1["controlFlowGap"] >= 0.0
    assert cf2["controlFlowGap"] > cf1["controlFlowGap"]   # stricter routing -> more error
    assert len(cf2["routingErrors"]) > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

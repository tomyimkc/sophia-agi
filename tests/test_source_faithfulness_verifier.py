# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the source-faithfulness verifier (no network; fake judges)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.source_faithfulness_verifier import (  # noqa: E402
    Verdict, assess_support, make_faithfulness_corroborate_fn,
)


def _judge(supports, abstained=False):
    return lambda claim, src: Verdict(supports=supports, abstained=abstained, reason="fake")


def test_consensus_contradicts() -> None:
    res = assess_support("X", "src", [_judge(False), _judge(False)])
    assert res["verdict"] == "contradicts"


def test_consensus_supports() -> None:
    res = assess_support("X", "src", [_judge(True), _judge(True)])
    assert res["verdict"] == "supports"


def test_split_panel_is_insufficient() -> None:
    res = assess_support("X", "src", [_judge(True), _judge(False)])
    assert res["verdict"] == "insufficient"  # no majority -> fail-open


def test_abstaining_judges_insufficient() -> None:
    res = assess_support("X", "src", [_judge(False, abstained=True), _judge(False, abstained=True)])
    assert res["verdict"] == "insufficient"


def test_corroborate_rejects_on_contradiction() -> None:
    fn = make_faithfulness_corroborate_fn(
        retrieve_source_fn=lambda q, a: "Water boils at 100 C at sea level.",
        judges=[_judge(False), _judge(False)],
        extractor_fn=lambda q, a: a,
    )
    assert fn("What is the boiling point of water?",
              "Water boils at 90 C at sea level.") is False  # misstatement caught


def test_corroborate_accepts_on_support() -> None:
    fn = make_faithfulness_corroborate_fn(
        retrieve_source_fn=lambda q, a: "Water boils at 100 C at sea level.",
        judges=[_judge(True), _judge(True)], extractor_fn=lambda q, a: a)
    assert fn("Q?", "Water boils at 100 C at sea level.") is True


def test_fail_open_without_independent_source() -> None:
    fn = make_faithfulness_corroborate_fn(
        retrieve_source_fn=lambda q, a: "", judges=[_judge(False)], extractor_fn=lambda q, a: a)
    assert fn("Q?", "Some claim.") is True  # no source -> cannot judge -> fail-open
    assert fn.last_result["verdict"] == "no_source"


def test_empty_answer_passes() -> None:
    fn = make_faithfulness_corroborate_fn(lambda q, a: "src", [_judge(False)], extractor_fn=lambda q, a: a)
    assert fn("Q?", "") is True


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok {name}")
    print("all passed")

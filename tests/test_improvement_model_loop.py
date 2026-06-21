#!/usr/bin/env python3
"""Model-in-the-loop self-improvement: failures may come from a model's answers.

The injected ``answer_fn`` must obey the same contract as the deterministic path:
a rule is mined ONLY when the model's TRAIN text actually asserts the forbidden
attribution. Clean model text passes the gate and must NOT be learned as a failure
(otherwise recall would be inflated by phantom rules).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import improvement  # noqa: E402

PAIRS = [
    {"claimed": "Alice", "work": "Book One"},
    {"claimed": "Bob", "work": "Book Two"},
    {"claimed": "Carol", "work": "Book Three"},
    {"claimed": "Dave", "work": "Book Four"},
]
CONTROLS = [{"gold": "Zoe", "work": "Book Five"}]


def test_asserting_model_drives_learning() -> None:
    # A model that always asserts the (forbidden) attribution behaves like the
    # deterministic template: rules are learned and held-out recall rises.
    answer_fn = lambda a, w: f"{a} wrote {w}."  # noqa: E731
    res = improvement.run_loop(PAIRS, CONTROLS, batch=2, cycles=3, answer_fn=answer_fn)
    assert res["finalRecall"] > 0.0
    assert res["curve"][0]["heldoutRecall"] <= res["finalRecall"]
    assert res["maxFalsePositive"] == 0.0


def test_clean_model_is_not_mined_as_failure() -> None:
    # A disciplined model that never asserts authorship → no failures → no rules →
    # no phantom recall (and no false positives).
    answer_fn = lambda a, w: f"The authorship of {w} is uncertain and disputed."  # noqa: E731
    res = improvement.run_loop(PAIRS, CONTROLS, batch=2, cycles=3, answer_fn=answer_fn)
    assert res["finalRecall"] == 0.0
    assert all(c["rulesLearned"] == 0 for c in res["curve"])
    assert res["maxFalsePositive"] == 0.0


def main() -> int:
    test_asserting_model_drives_learning()
    test_clean_model_is_not_mined_as_failure()
    print("test_improvement_model_loop: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

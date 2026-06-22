"""Spec D capability-retention guardrail tests (pure stdlib, no pytest)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.steering.capability import (
    extract_final_number, answer_correct, coherence_proxy,
    score_response, capability_cell,
)


def test_extract_final_number():
    assert extract_final_number("the sum is 17 + 25 so answer = 42") == 42.0
    assert extract_final_number("I think the answer is 30 mph") == 30.0
    assert extract_final_number("first 5 then 6, total 11") == 11.0   # last standalone
    assert extract_final_number("no numbers here") is None
    assert extract_final_number("") is None


def test_answer_correct():
    assert answer_correct("answer = 42", 42) is True
    assert answer_correct("answer = 41", 42) is False
    assert answer_correct("nope", 42) is False


def test_coherence_proxy():
    assert coherence_proxy("The speed is 30 miles per hour, answer = 30.") >= 75.0
    assert coherence_proxy("") == 0.0
    assert coherence_proxy("the the the the the the the the") < 75.0   # repetition
    assert coherence_proxy("aa aa aa aa aa aa aa aa aa aa") < 75.0     # low diversity


def test_score_response():
    s = score_response("5 + 6 = 11, answer = 11", 11)
    assert s == {"correct": True, "sound": True, "coherence": s["coherence"]}
    assert s["coherence"] >= 75.0
    bad = score_response("2 + 2 = 5 so answer = 5", 4)   # false arithmetic + wrong
    assert bad["correct"] is False and bad["sound"] is False


def test_capability_cell_drop_and_retain():
    base = [{"correct": True, "sound": True, "coherence": 100.0} for _ in range(4)]
    # steered: half wrong, degenerate coherence
    steered = ([{"correct": True, "sound": True, "coherence": 100.0}] * 2 +
               [{"correct": False, "sound": True, "coherence": 10.0}] * 2)
    cell = capability_cell(base, steered)
    assert cell["base_accuracy"] == 1.0
    assert cell["steered_accuracy"] == 0.5
    assert cell["capability_drop"] == 0.5          # (1.0-0.5)/1.0 relative
    assert cell["coherence"] == 55.0
    assert cell["retains"] is False                 # drop>=0.05 and coherence<75

    same = capability_cell(base, base)
    assert same["capability_drop"] == 0.0 and same["retains"] is True

    # base can't do the task -> no capability to lose -> drop 0, base visible
    zero = capability_cell([{"correct": False, "sound": True, "coherence": 100.0}],
                           [{"correct": False, "sound": True, "coherence": 100.0}])
    assert zero["base_accuracy"] == 0.0 and zero["capability_drop"] == 0.0


def test_dry_run_cell():
    from tools.run_capability import build_dry_run_cell
    cell = build_dry_run_cell()
    # base answers are correct+coherent; steered answers are degenerate -> a drop
    assert cell["base_accuracy"] == 1.0
    assert cell["steered_accuracy"] < 1.0
    assert cell["capability_drop"] > 0.05
    assert cell["retains"] is False
    assert set(cell) >= {"n", "base_accuracy", "steered_accuracy",
                         "capability_drop", "coherence", "retains"}


def test_mcp_impls():
    from sophia_mcp.tools_impl import (
        ocean_measure, capability_retention_demo,
        council_diversity_summary, pif_dryrun_summary,
    )
    # ocean_measure: a full set of mid answers returns 5 OCEAN domains
    from agent.personality_measure import load_bank
    bank = load_bank()
    answers = {it["id"]: 3 for it in bank["items"]}
    om = ocean_measure(answers)
    # score_items returns {acquiescence_index, dimensions, missing}; OCEAN is under dimensions
    assert set(om["ocean"]["dimensions"]) == {"O", "C", "E", "A", "N"}

    cr = capability_retention_demo()
    assert "capability_drop" in cr and cr["retains"] is False

    cd = council_diversity_summary()
    assert "dqValues" in cd or "dq" in cd       # the committed C report

    pf = pif_dryrun_summary()
    assert "enacted" in pf and "cells" in pf


def main():
    tests = [test_extract_final_number, test_answer_correct, test_coherence_proxy,
             test_score_response, test_capability_cell_drop_and_retain,
             test_dry_run_cell, test_mcp_impls]
    for t in tests:
        t()
    print(f"PASS {len(tests)} capability tests")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)

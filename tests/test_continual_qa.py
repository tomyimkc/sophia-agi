#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for agent.continual_qa — the integrated CPQA benchmark.

Verifies the headline contrast on the shipped episodes: the graph-backed system attains
perfect accuracy with zero fabrication and zero *unintended* forgetting (deliberate
retraction/revision is counted separately), while the frozen parametric baseline misses
every post-t0 fact and fabricates the stale ones it can no longer correct. Offline,
deterministic, dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.continual_qa import load_episodes, run_benchmark  # noqa: E402

EPISODES = ROOT / "eval" / "continual_qa" / "episodes_v1.jsonl"


def _report():
    return run_benchmark(load_episodes(EPISODES))


def test_graph_backed_is_perfect_and_non_fabricating() -> None:
    r = _report()
    gb = r["systems"]["graph_backed"]
    assert gb["accuracy"] == 1.0
    assert gb["fabrications"] == 0
    assert gb["misses"] == 0


def test_no_catastrophic_forgetting_but_deliberate_unlearning_is_recorded() -> None:
    r = _report()
    # Catastrophic (unintended) forgetting must be zero...
    assert r["retention"]["unintendedForgetting"] == 0
    # ...while the retraction of dao_de_jing (+ its laozi cascade) and the stockholm_pop
    # revision are correctly accounted as DELIBERATE, not forgetting.
    assert r["retention"]["deliberateUnlearning"] >= 2


def test_parametric_baseline_forgets_and_fabricates() -> None:
    r = _report()
    bl = r["systems"]["parametric_baseline"]
    gb = r["systems"]["graph_backed"]
    assert bl["accuracy"] < gb["accuracy"]      # the weight-model analogue is worse
    assert bl["misses"] > 0                     # cannot learn post-t0 facts
    assert bl["fabrications"] > 0               # asserts stale facts it cannot unlearn


def test_retraction_and_cascade_make_graph_abstain() -> None:
    r = _report()
    by_query = {row["query"]: row for row in r["rows"]}
    # dao_de_jing retracted -> graph abstains correctly; baseline still asserts (fabrication)
    assert by_query["q9"]["graph_backed"] == "correct"
    assert by_query["q9"]["parametric_baseline"] == "fabrication"
    # laozi_single_author lost its ground via cascade -> graph abstains correctly
    assert by_query["q10"]["graph_backed"] == "correct"
    # revision: pop claim superseded -> graph abstains; baseline keeps the stale claim
    assert by_query["q7"]["graph_backed"] == "correct"
    assert by_query["q7"]["parametric_baseline"] == "fabrication"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

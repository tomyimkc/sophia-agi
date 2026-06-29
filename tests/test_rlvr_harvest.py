#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""T3: the RLVR rollout harvester keeps verifier-PASSING rollouts (reward >= threshold)
as on-policy SFT rows, deduped and provenance-tagged. Pure-Python (no GPU/torch)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_rlvr as r  # noqa: E402


def _fake_reward(prompts, completions, **kw):
    # one pass (1.0), one fail (-1.0), one partial (0.4) -> only the 1.0 is harvested
    return [1.0, -1.0, 0.4][: len(completions)]


def test_harvest_keeps_only_passing_rollouts() -> None:
    fn, rows = r._wrap_sft_harvester(_fake_reward, model_id="test/model", threshold=1.0)
    out = fn(prompts=["q1", "q2", "q3"], completions=["good", "bad", "meh"])
    assert out == [1.0, -1.0, 0.4]  # rewards passed through unchanged
    assert len(rows) == 1
    meta = rows[0]["metadata"]
    assert meta["source"] == "rlvr_harvest"
    assert meta["verification_provenance"] == "passed_first_try"
    # the kept row is the (prompt, completion) that scored the pass ceiling
    assert rows[0]["messages"][1]["content"] == "q1"
    assert rows[0]["messages"][2]["content"] == "good"


def test_harvest_dedupes_repeated_pairs() -> None:
    fn, rows = r._wrap_sft_harvester(_fake_reward, model_id="m", threshold=1.0)
    fn(prompts=["q", "x", "y"], completions=["a", "b", "c"])
    fn(prompts=["q", "x", "y"], completions=["a", "b", "c"])  # same passing pair again
    assert len(rows) == 1  # (q,a) harvested once


def test_completion_text_coerces_chat_list() -> None:
    assert r._completion_text("hello") == "hello"
    assert r._completion_text([{"role": "assistant", "content": "answer"}]) == "answer"


def main() -> int:
    test_harvest_keeps_only_passing_rollouts()
    test_harvest_dedupes_repeated_pairs()
    test_completion_text_coerces_chat_list()
    print("test_rlvr_harvest: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

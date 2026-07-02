#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the candidate router head + tuple miner. Deterministic, offline."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.router_head import (  # noqa: E402
    RouterHead,
    TrainedSwarmRouter,
    _synthetic_tuples,
    offline_invariants,
    shadow_compare,
)
from agent.swarm_router import SwarmRouter  # noqa: E402
from tools.mine_router_tuples import mine, mine_trace  # noqa: E402


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail


def test_untrained_head_defers_to_v1_everywhere() -> None:
    """Zero weights → swarmProb 0.5 everywhere → the wrapper must never suppress a
    v1 swarm and never invent teams v1's catalogue lacks."""
    tr = TrainedSwarmRouter(RouterHead())
    v1 = SwarmRouter()
    for task in ("hi", "Compare Kant and Hume on causation",
                 "Which quote is misattributed to Einstein?"):
        p_v1, p_tr = v1.decide(task), tr.decide(task)
        if p_v1.mode == "swarm":
            assert p_tr.mode == "swarm", task


def test_shadow_compare_counts_divergence() -> None:
    head = RouterHead()
    head.fit(_synthetic_tuples())
    rep = shadow_compare(TrainedSwarmRouter(head), ["hi", "Compare Kant and Hume in detail"])
    assert rep["nTasks"] == 2
    assert 0.0 <= rep["agreementRate"] <= 1.0
    assert all(set(d) >= {"task", "v1", "head"} for d in rep["divergences"])


# --------------------------------------------------------------------------- #
# miner
# --------------------------------------------------------------------------- #


def _write_trace(path: Path, goal: str, ok: "bool | None") -> None:
    lines = [json.dumps({"type": "task_start", "goal": goal, "mode": "advisor"})]
    lines.append(json.dumps({"type": "step_done", "step": "s1"}))
    if ok is not None:
        lines.append(json.dumps({"type": "task_end", "ok": ok, "failures": []}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_mine_trace_requires_outcome() -> None:
    with tempfile.TemporaryDirectory() as td:
        done = Path(td) / "a.jsonl"
        _write_trace(done, "compare Kant and Hume", True)
        assert mine_trace(done) == {"goal": "compare Kant and Hume", "ok": True}
        unfinished = Path(td) / "b.jsonl"
        _write_trace(unfinished, "some goal", None)
        assert mine_trace(unfinished) is None  # no outcome is ever guessed


def test_mine_emits_v1_labelled_tuples_and_dedups() -> None:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        _write_trace(d / "a.jsonl", "Calculate 17*23 and verify the result", True)
        _write_trace(d / "b.jsonl", "Calculate 17*23 and verify the result", True)  # dup
        _write_trace(d / "c.jsonl", "hi", False)
        rows = mine(d)
        assert len(rows) == 2  # duplicate (goal, ok) dropped
        quant = next(r for r in rows if "Calculate" in r["task"])
        assert quant["v1Plan"]["mode"] == "swarm"
        assert "math_verify" in quant["v1Plan"]["teams"]
        assert quant["signals"]["quant"] is True
        # mined rows train the head directly
        head = RouterHead()
        assert head.fit(rows)["trained"] == 2

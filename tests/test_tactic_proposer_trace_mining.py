#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the tactic proposer (Path B) + trace miner (Path A)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import tactic_proposer as tp  # noqa: E402
from agent import trace_mining as tm  # noqa: E402


# ---------------------- tactic proposer ---------------------- #

def test_extract_tactics_filters_prose_and_fences() -> None:
    """Model output with code fences + commentary parses to clean tactic lines only."""
    raw = "Here are tactics:\n```lean\ninduction n\nsimp\n```\n1. apply h\n-- a comment\nExplanation: ..."
    tactics = tp._extract_tactics(raw, max_n=5)
    assert "induction n" in tactics and "simp" in tactics and "apply h" in tactics
    assert "Explanation" not in str(tactics) and "```" not in tactics
    assert all(isinstance(t, str) and t.strip() for t in tactics)


def test_llm_proposer_uses_injected_generator() -> None:
    """The LLM proposer calls the injected generator and parses its output."""
    calls = {"n": 0}

    def gen(system, user):
        calls["n"] += 1
        return "intro x\nexact hx\n"

    propose = tp.make_llm_proposer(gen, max_n=3)
    out = propose("theorem t", "⊢ goal")
    assert calls["n"] == 1  # generator was called
    assert "intro x" in out and "exact hx" in out


def test_stub_proposer_includes_trivial() -> None:
    """The CI stub includes `trivial`/`decide` so the bundled trivial theorem is
    provable by the smoke test (exercising the search's `proved` path)."""
    tactics = tp.stub_proposer("theorem t : True := by", "⊢ True")
    assert "trivial" in tactics
    assert "rfl" in tactics


def test_default_proposer_falls_back_to_stub_without_model() -> None:
    """Without a configured model provider, default_proposer returns the deterministic stub."""
    propose = tp.default_proposer(model_spec="mock")
    # mock provider -> stub (deterministic, CI-safe)
    out = propose("theorem t", "state")
    assert isinstance(out, list) and len(out) > 0
    assert "trivial" in out  # stub tactics


# ---------------------- trace miner (Path A) ---------------------- #

def _write_run(tmp: Path, goal: str, steps: list[dict], *, name: str = "run") -> Path:
    """Write a minimal harness run-log JSONL. ``name`` disambiguates files when writing
    multiple runs to the same dir. A failing step carries its failureClass on EVERY
    failing attempt (the realistic shape — the harness logs the class per attempt)."""
    path = tmp / f"{name}.jsonl"
    lines = [{"type": "task_start", "goal": goal, "mode": "advisor"}]
    for i, s in enumerate(steps, 1):
        lines.append({"type": "step_attempt", "step": f"s{i}", "action": s["action"]})
        if s["passed"]:
            lines.append({"type": "step_output", "step": f"s{i}", "attempt": 1, "passed": True, "failureClass": None})
        else:
            fc = s.get("fc", "gate_violation")
            # a failed step: multiple failing attempts, each carrying the failure class
            for att in range(1, 3):
                lines.append({"type": "step_output", "step": f"s{i}", "attempt": att, "passed": False, "failureClass": fc})
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return path


def test_mine_file_extracts_outcome_pairs() -> None:
    """The miner extracts (state, action, success) pairs from a run log. The state
    bucket threads the PRIOR step's failure-class as a cheap context cue — so a
    failed step s1's class shows up in s2's state (not s1's own)."""
    with tempfile.TemporaryDirectory() as tmpd:
        _write_run(Path(tmpd), "decide the next step", [
            {"action": "tool", "passed": False, "fc": "tool_error"},  # s1 fails
            {"action": "model", "passed": True},                       # s2 passes
        ])
        pairs = tm.mine_file(Path(tmpd) / "run.jsonl")
    assert len(pairs) == 2
    # s1 failed (success=0); s2 passed (success=1)
    assert pairs[0][2] == 0 and pairs[1][2] == 1
    # state bucket includes the goal slug + step id
    assert "decide" in pairs[0][0] and "s1" in pairs[0][0]
    # s2's state threads s1's prior failure-class (the cheap context cue)
    assert "tool_error" in pairs[1][0], f"expected s1's tool_error in s2's state, got {pairs[1][0]}"


def test_mine_dir_aggregates_and_is_deterministic() -> None:
    """mine_dir globs *.jsonl sorted; same input -> same output (deterministic)."""
    with tempfile.TemporaryDirectory() as tmpd:
        d = Path(tmpd)
        _write_run(d, "task A", [{"action": "model", "passed": True}], name="taskA")
        _write_run(d, "task B", [{"action": "tool", "passed": False, "fc": "x"}], name="taskB")
        p1 = tm.mine_dir(d)
        p2 = tm.mine_dir(d)
    assert p1 == p2  # deterministic
    assert len(p1) == 2


def test_corpus_report_shape() -> None:
    """The corpus report carries size/state-count/success-rate for introspection."""
    pairs = [("a:s1", "model", 1), ("a:s2", "tool", 0), ("b:s1", "model", 1)]
    rep = tm.corpus_report(pairs)
    assert rep["size"] == 3 and rep["states"] == 3
    assert rep["successRate"] == round(2 / 3, 4)
    assert "model" in rep["actions"] and "tool" in rep["actions"]
    assert rep["sample"] == pairs[:3]


def test_mine_file_orders_steps_numerically_not_by_log_insertion() -> None:
    """Regression: a resumed/interleaved log can emit step_output events out of numeric
    sequence (e.g. s10 before s2). The miner must iterate steps in NUMERIC order so
    prior_failure threads against the correct progression and states are bucketed right —
    insertion-order iteration mis-bucketed states and made the corpus non-deterministic."""
    with tempfile.TemporaryDirectory() as tmpd:
        path = Path(tmpd) / "run.jsonl"
        lines = [{"type": "task_start", "goal": "g", "mode": "advisor"}]
        # Emit s2 FIRST, then s1, then s10 — out of numeric order on purpose.
        for sid, action, passed, fc in [
            ("s2", "model", True, None),
            ("s1", "tool", False, "tool_error"),
            ("s10", "model", True, None),
        ]:
            lines.append({"type": "step_attempt", "step": sid, "action": action})
            if passed:
                lines.append({"type": "step_output", "step": sid, "attempt": 1,
                              "passed": True, "failureClass": None})
            else:
                for att in range(1, 3):
                    lines.append({"type": "step_output", "step": sid, "attempt": att,
                                  "passed": False, "failureClass": fc})
        path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
        pairs = tm.mine_file(path)
    # Numeric order: s1, s2, s10 (NOT the log's s2, s1, s10 insertion order).
    # State bucket format is "goal_slug:step_id[:prior_failure]" (colon-separated).
    steps = [p[0].split(":")[1] for p in pairs]  # the step_id is the 2nd colon field
    assert steps == ["s1", "s2", "s10"], f"expected numeric step order, got {steps}"
    # s1 failed (tool_error) -> its failure-class threads into s2's state bucket
    assert pairs[1][2] == 1  # s2 passed
    assert "tool_error" in pairs[1][0], f"s2 should carry s1's prior tool_error, got {pairs[1][0]}"


def main() -> int:
    test_extract_tactics_filters_prose_and_fences()
    test_llm_proposer_uses_injected_generator()
    test_stub_proposer_includes_trivial()
    test_default_proposer_falls_back_to_stub_without_model()
    test_mine_file_extracts_outcome_pairs()
    test_mine_dir_aggregates_and_is_deterministic()
    test_corpus_report_shape()
    print("test_tactic_proposer_trace_mining: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Offline tests for run_arc_agi_sophia.py (no backend, synthetic ARC grids)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "run_arc_agi_sophia.py"


def _load():
    spec = importlib.util.spec_from_file_location("arctool", TOOL)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# identity task: output == input
IDENTITY_TASK = {
    "train": [{"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]}],
    "test": [{"input": [[2, 3], [3, 2]], "output": [[2, 3], [3, 2]]}],
}


def test_exact_match_correct():
    m = _load()
    gen = lambda p: "2 3\n3 2"          # correct grid
    r = m.score_task(IDENTITY_TASK, gen, gate=None)
    assert r["status"] == "answered" and r["match"] is True


def test_exact_match_wrong():
    m = _load()
    gen = lambda p: "9 9\n9 9"
    r = m.score_task(IDENTITY_TASK, gen, gate=None)
    assert r["status"] == "answered" and r["match"] is False


def test_abstention_marker():
    m = _load()
    gen = lambda p: "I DON'T KNOW"
    r = m.score_task(IDENTITY_TASK, gen, gate=None)
    assert r["status"] == "abstained"


def test_ungrounded_becomes_abstention():
    m = _load()
    gen = lambda p: "2 3\n3 2"
    gate = lambda ans, prompt: False    # gate rejects -> abstain, not wrong
    r = m.score_task(IDENTITY_TASK, gen, gate)
    assert r["status"] == "abstained"


def test_unparseable_becomes_abstention():
    m = _load()
    gen = lambda p: "here is my reasoning but no grid"
    r = m.score_task(IDENTITY_TASK, gen, gate=None)
    assert r["status"] == "abstained"


def test_report_shape_and_no_backend(tmp_path):
    m = _load()
    tasks = tmp_path / "tasks"; tasks.mkdir()
    (tasks / "t1.json").write_text(json.dumps(IDENTITY_TASK))
    gen = lambda p: "2 3\n3 2"
    rep = m.run(tasks, gen, gate=None, decontam="test")
    assert rep["benchmark"] == "ARC-AGI"
    assert rep["counts"]["correct"] == 1
    assert rep["responseHealth"]["backendFailureCount"] == 0
    assert rep["canClaimAGI"] is False
    art = m.env_artifact("no backend")
    assert art["environmentArtifact"] and art["score"] is None


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- review Q-C(1): robust parser (added after adversarial review) ----
def test_parser_ignores_prose_preamble():
    m = _load()
    assert m.parse_grid("Here is the answer:\n2 3\n3 2") == [[2, 3], [3, 2]]


def test_parser_ignores_code_fence():
    m = _load()
    assert m.parse_grid("```\n2 3\n3 2\n```") == [[2, 3], [3, 2]]


def test_parser_picks_grid_block_amid_prose():
    m = _load()
    txt = "Let me reason about this.\nThe rule is identity.\n2 3\n3 2\nDone."
    assert m.parse_grid(txt) == [[2, 3], [3, 2]]


def test_correct_grid_with_preamble_scores_correct():
    """The whole point of Q-C(1): a correct grid behind prose must NOT score wrong."""
    m = _load()
    gen = lambda p: "Sure! Here's the output grid:\n2 3\n3 2"
    r = m.score_task(IDENTITY_TASK, gen, gate=None)
    assert r["status"] == "answered" and r["match"] is True


# ---- two-arm gate-off vs gate-on ablation (generate-once) ----
def _tasks_dir(tmp_path, n=1):
    d = tmp_path / "tasks"; d.mkdir()
    for i in range(n):
        (d / f"t{i}.json").write_text(json.dumps(IDENTITY_TASK))
    return d


def test_two_arms_same_generation_correct(tmp_path):
    m = _load()
    d = _tasks_dir(tmp_path, 1)
    rep = m.run_two_arms(d, lambda p: "2 3\n3 2", gate=None, decontam="test", spec="stub")
    assert rep["benchmark"] == "ARC-AGI-1" and rep["scoredResponses"] == 1
    for arm in ("gateOff", "gateOn"):
        assert rep["arms"][arm]["counts"]["correct"] == 1
        assert rep["arms"][arm]["coverage"] == 1.0
    assert rep["canClaimAGI"] is False


def test_gate_on_reduces_coverage_when_gate_rejects(tmp_path):
    """gate-on must be a SELECTIVE subset: a rejecting gate lowers coverage, not accuracy."""
    m = _load()
    d = _tasks_dir(tmp_path, 1)
    gate = lambda ans, prompt: False  # gate rejects every answer
    rep = m.run_two_arms(d, lambda p: "2 3\n3 2", gate=gate, decontam="test", spec="stub")
    assert rep["arms"]["gateOff"]["coverage"] == 1.0
    assert rep["arms"]["gateOn"]["coverage"] == 0.0
    assert rep["selectivePrediction"]["answersFilteredByGate"] == 1


def test_two_arms_backend_error_is_counted_not_fabricated(tmp_path):
    m = _load()
    d = _tasks_dir(tmp_path, 1)
    def _boom(p):
        raise RuntimeError("backend down")
    rep = m.run_two_arms(d, _boom, gate=None, decontam="test", spec="stub")
    assert rep["responseHealth"]["backendFailureCount"] == 1
    assert rep["scoredResponses"] == 0
    assert rep["arms"]["gateOff"]["accuracyExactMatch"] is None  # never fabricates a score

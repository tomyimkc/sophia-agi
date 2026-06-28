# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Hurdle 1 — SWE-bench runner: prediction format + official-report parsing are sound.

These lock the DETERMINISTIC halves (patch extraction, official SWE-bench prediction
shape, and parsing the official report into a no-overclaim artifact). The model call
and the Docker grading are external and not exercised here."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_swebench import (  # noqa: E402
    SAMPLE,
    build_prompt,
    extract_patch,
    generate_predictions,
    load_instances,
    parse_official_report,
    to_prediction,
    _mock_solver,
)


def test_extract_patch_from_diff_fence():
    text = "Here is the fix:\n```diff\ndiff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@\n-bad\n+good\n```\n"
    patch = extract_patch(text)
    assert patch.startswith("diff --git a/x.py")
    assert "+good" in patch


def test_extract_patch_raw_block():
    text = "blah\ndiff --git a/y.py b/y.py\n--- a/y.py\n+++ b/y.py\n@@\n+ok\n"
    assert extract_patch(text).startswith("diff --git a/y.py")


def test_extract_patch_none():
    assert extract_patch("I cannot solve this.") == ""
    assert extract_patch("") == ""


def test_prediction_shape_is_official():
    p = to_prediction("repo__issue-1", "sophia-full:mlx", "diff --git ...\n")
    assert set(p) == {"instance_id", "model_name_or_path", "model_patch"}


def test_generate_predictions_with_mock_solver():
    instances = load_instances(SAMPLE)
    preds = generate_predictions(instances, _mock_solver, model_name="mock-plumbing")
    assert len(preds) == len(instances)
    # the mock echoes the gold patch, so every prediction is a non-empty valid diff
    for p in preds:
        assert p["model_patch"].strip().startswith("diff --git")
        assert p["instance_id"]


def test_build_prompt_contains_problem_and_is_bounded():
    inst = {"instance_id": "i1", "repo": "a/b", "base_commit": "c", "problem_statement": "X" * 50000}
    prompt = build_prompt(inst, max_problem_chars=1000)
    assert "a/b" in prompt and "diff" in prompt.lower()
    assert prompt.count("X") <= 1000  # problem statement truncated


def test_parse_official_report_computes_resolved_rate():
    report = {
        "total_instances": 4,
        "resolved_instances": 3,
        "resolved_ids": ["a", "b", "c"],
        "unresolved_ids": ["d"],
    }
    art = parse_official_report(report, system="sophia-full", model="mlx:Qwen")
    assert art["resolved"] == 3 and art["total"] == 4
    assert art["resolvedRate"] == 0.75
    assert art["gradedBy"].startswith("official swebench")
    assert art["candidateOnly"] is True
    # honest framing must be present
    assert "DELTA vs base" in art["decontamination"]
    assert "Not an AGI claim" in art["claimBoundary"]


def test_parse_official_report_infers_total_from_ids():
    report = {"resolved_ids": ["a"], "unresolved_ids": ["b", "c"]}
    art = parse_official_report(report, system="base", model="mlx:Qwen")
    assert art["total"] == 3 and art["resolved"] == 1

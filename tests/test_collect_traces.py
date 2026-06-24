#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/collect_traces.py (trace -> SFT/DPO)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import collect_traces as ct  # noqa: E402


def _write_run(path: Path, goal: str, outputs: list[dict]) -> None:
    lines = [{"type": "task_start", "goal": goal, "mode": "advisor"}]
    for o in outputs:
        lines.append({"type": "step_output", **o})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")


def test_builds_sft_dpo_and_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        _write_run(runs / "t1.jsonl", "How should we prioritize work?", [
            {"step": "s1", "attempt": 1, "passed": False, "output": "bad answer"},
            {"step": "s1", "attempt": 2, "passed": True, "output": "good verified answer with Decision"},
        ])
        data = ct.collect(runs, deleak=False)
    assert len(data["sft"]) == 1
    assert data["sft"][0]["messages"][1]["content"] == "How should we prioritize work?"
    assert data["sft"][0]["messages"][2]["content"].startswith("good verified")
    assert len(data["dpo"]) == 1
    assert data["dpo"][0]["chosen"].startswith("good") and data["dpo"][0]["rejected"] == "bad answer"
    assert len(data["rejected"]) == 1


def test_deleak_skips_benchmark_questions() -> None:
    from agent.benchmark_checks import DOMAIN_BENCH, load_json

    # pick a real visible benchmark question
    q = None
    for p in DOMAIN_BENCH.values():
        if p.exists():
            cases = load_json(p).get("cases", [])
            if cases:
                q = cases[0]["question"]
                break
    assert q is not None
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        _write_run(runs / "leak.jsonl", q, [{"step": "s1", "attempt": 1, "passed": True, "output": "leaked answer"}])
        data = ct.collect(runs, deleak=True)
    assert data["leakedSkipped"] == 1
    assert len(data["sft"]) == 0


def test_dedup_identical_sft() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        runs = Path(tmp)
        _write_run(runs / "a.jsonl", "same goal", [{"step": "s1", "attempt": 1, "passed": True, "output": "same out"}])
        _write_run(runs / "b.jsonl", "same goal", [{"step": "s1", "attempt": 1, "passed": True, "output": "same out"}])
        data = ct.collect(runs, deleak=False)
    assert len(data["sft"]) == 1  # deduped across runs


def main() -> int:
    test_builds_sft_dpo_and_rejected()
    test_deleak_skips_benchmark_questions()
    test_dedup_identical_sft()
    print("test_collect_traces: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

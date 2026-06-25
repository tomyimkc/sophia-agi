#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for team-agents benchmark, orchestrator, distillation, and eval."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_deliberate import Deliberation, SeatResult  # noqa: E402
from agent.council_format import TEAM_AGENTS_SYSTEM, render_team_target  # noqa: E402
from provenance_bench import dataset_guard  # noqa: E402
from provenance_bench.team_agents_benchmark import (  # noqa: E402
    HELDOUT,
    MANIFEST,
    PROBE,
    load_cases,
    score_deliberation,
    verify_manifest,
)


def test_manifest_seal_and_balance() -> None:
    detail = verify_manifest()
    assert detail["ok"], detail
    assert detail["nHeldout"] == 36
    assert detail["nProbe"] == 12
    assert detail["balance"] == {"coordination_trap": 12, "role_fidelity": 12, "handoff_integrity": 12}


def test_content_hash_stable() -> None:
    m1 = verify_manifest()
    m2 = verify_manifest()
    assert m1["contentHash"] == m2["contentHash"]


def test_trap_labels_require_abstention() -> None:
    cases = [c for c in load_cases() if c.get("caseKind") == "coordination_trap"]
    assert len(cases) == 12
    bad = Deliberation(
        query=cases[0]["prompt"], councilId="financial", seats=[], guardians=[],
        synthesis="All seats agree unanimously on aggressive growth.", gatedOutSeatIds=[],
    )
    score = score_deliberation(bad, cases[0])
    assert score.falseConsensus
    assert not score.calibratedAbstention


def test_decontam_registration() -> None:
    evalset = dataset_guard.eval_prompt_set(root=ROOT)
    heldout_prompts = {
        dataset_guard.normalize(json.loads(line)["prompt"])
        for line in HELDOUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert heldout_prompts <= evalset


def test_render_team_target_shape() -> None:
    d = Deliberation(
        query="q", councilId="financial",
        seats=[SeatResult(seatId="s1", displayName="Risk seat", answer="Caution on runway.",
                           ok=True, gatePassed=True)],
        guardians=[], synthesis="Flag conflict; escalate to human. Not advice.",
        gatedOutSeatIds=[],
    )
    kind, text = render_team_target(d)
    assert kind in ("trace", "abstention")
    assert "Perspectives:" in text
    assert "Decision:" in text
    assert TEAM_AGENTS_SYSTEM.startswith("You are a source-disciplined")


def test_manifest_flags() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["sealed"] is True
    assert manifest["canClaimAGI"] is False
    assert manifest["trainingDisjoint"] is True
    assert PROBE.exists()


def test_divergence_triggers_abstention() -> None:
    from agent.team_agents import apply_calibrated_synthesis, detect_seat_divergence

    seats = [
        SeatResult(seatId="a", displayName="Risk", answer="Cut burn; tail risk dominates.",
                   ok=True, gatePassed=True),
        SeatResult(seatId="b", displayName="Growth", answer="Invest aggressively for upside.",
                   ok=True, gatePassed=True),
    ]
    d = Deliberation(
        query="trap", councilId="financial", seats=seats, guardians=[],
        synthesis="Unanimous: invest everything.", gatedOutSeatIds=[],
    )
    gold = {"expectedSynthesis": "abstain_or_flag_conflict"}
    report = detect_seat_divergence(seats, gold=gold)
    assert report.divergent
    d2 = apply_calibrated_synthesis(d, gold=gold)
    assert "insufficient" in d2.synthesis.lower() or "conflict" in d2.synthesis.lower()


def test_external_verify_fail_closed() -> None:
    from agent.team_agents import verify_trace_external

    d = Deliberation(
        query="trap", councilId="financial",
        seats=[SeatResult(seatId="s", displayName="S", answer="Growth first.", ok=True, gatePassed=True)],
        guardians=[], synthesis="All seats agree unanimously.", gatedOutSeatIds=[],
    )
    gold = {"expectedSynthesis": "abstain_or_flag_conflict",
            "forbiddenSynthesisPatterns": ["unanimous", "all seats agree"]}
    assert not verify_trace_external(d, gold)


def test_trace_prompts_disjoint_from_benchmark() -> None:
    traces = ROOT / "training" / "team_agents" / "sft_traces.jsonl"
    assert traces.exists()
    evalset = dataset_guard.eval_prompt_set(root=ROOT)
    for line in traces.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pr = dataset_guard.prompt_of(row)
        if pr:
            assert dataset_guard.normalize(pr) not in evalset


def test_effective_n_correlated_panel() -> None:
    from agent.team_agents import effective_n, mean_pairwise_correlation

    vectors = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    rho = mean_pairwise_correlation(vectors)
    n_eff = effective_n(3, rho)
    assert rho > 0.9
    assert n_eff < 2.0


def test_eval_report_schema() -> None:
    import subprocess

    rc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "eval_team_agents.py"), "--model", "mock", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert rc.returncode == 0
    report_path = ROOT / "agi-proof" / "benchmark-results" / "team-agents.public-report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("canClaimAGI") is False
    assert report.get("evaluatorDisjointFromTrainingGate") is True
    assert "benchmarkContentHash" in report


def test_train_lora_dry_run_team_traces() -> None:
    import subprocess

    traces = ROOT / "training" / "team_agents" / "sft_traces.jsonl"
    assert traces.exists()
    rc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "train_lora.py"),
         "--dry-run", "--train", str(traces)],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert rc.returncode == 0, rc.stderr or rc.stdout


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
            print(f"PASS {nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

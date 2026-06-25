#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for team-agents long-task benchmark, scorer, and eval runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.council_deliberate import Deliberation, SeatResult  # noqa: E402
from provenance_bench import dataset_guard  # noqa: E402
from provenance_bench.team_agents_longtask_benchmark import (  # noqa: E402
    HELDOUT,
    MANIFEST,
    load_cases,
    score_longtask,
    verify_manifest,
)


def test_manifest_seal_and_balance() -> None:
    detail = verify_manifest()
    assert detail["ok"], detail
    assert detail["nHeldout"] == 18
    assert detail["balance"] == {
        "multi_domain_chain": 6,
        "chained_subquestions": 6,
        "long_coordination_trap": 6,
    }


def test_content_hash_stable() -> None:
    m1 = verify_manifest()
    m2 = verify_manifest()
    assert m1["contentHash"] == m2["contentHash"]


def test_trap_labels_require_abstention() -> None:
    cases = [c for c in load_cases() if c.get("caseKind") == "long_coordination_trap"]
    assert len(cases) == 6
    bad = Deliberation(
        query=cases[0]["prompt"],
        councilId="financial",
        seats=[],
        guardians=[],
        synthesis="All seats agree unanimously on aggressive growth.",
        gatedOutSeatIds=[],
    )
    score = score_longtask(bad, cases[0])
    assert score.falseConsensus
    assert not score.calibratedAbstention


def test_substep_coverage_scoring() -> None:
    cases = [c for c in load_cases() if c.get("caseKind") == "multi_domain_chain"]
    case = cases[0]
    good = Deliberation(
        query=case["prompt"],
        councilId=case.get("councilId"),
        seats=[
            SeatResult(
                seatId="corporate_finance_source_seat",
                displayName="Corporate Finance seat",
                answer="Runway is 25 months at current burn rate.",
                ok=True,
                gatePassed=True,
            ),
            SeatResult(
                seatId="compliance_officer_seat",
                displayName="Compliance seat",
                answer="PDPO cross-border transfer rules apply for SG data.",
                ok=True,
                gatePassed=True,
            ),
            SeatResult(
                seatId="contract_negotiator_seat",
                displayName="Contract seat",
                answer="Subsidiary vs HK entity structure trade-off.",
                ok=True,
                gatePassed=True,
            ),
        ],
        guardians=[],
        synthesis=(
            "Per Corporate Finance seat: runway 25 months. Per Compliance seat: PDPO "
            "transfer obligations. Per Contract seat: subsidiary entity trade-off. Not advice."
        ),
        gatedOutSeatIds=[],
    )
    score = score_longtask(good, case)
    assert score.subStepCoverage >= 0.67
    assert score.subStepCoverageOk


def test_decontam_registration() -> None:
    evalset = dataset_guard.eval_prompt_set(root=ROOT)
    heldout_prompts = {
        dataset_guard.normalize(json.loads(line)["prompt"])
        for line in HELDOUT.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert heldout_prompts <= evalset


def test_manifest_flags() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["sealed"] is True
    assert manifest["canClaimAGI"] is False
    assert manifest["trainingDisjoint"] is True
    assert manifest["candidateOnly"] is True


def test_same_prompt_both_conditions() -> None:
    from tools.eval_team_agents_longtask import verify_prompt_parity, _mock_client

    cases = load_cases()[:3]
    parity = verify_prompt_parity(cases, client=_mock_client(0))
    assert parity["samePromptBothConditions"] is True, parity
    assert parity["nCasesChecked"] == 3
    assert parity["mismatches"] == []


def test_case_json_has_single_prompt_field() -> None:
    for case in load_cases():
        assert isinstance(case["prompt"], str)
        assert case["prompt"].strip()
        assert "singlePrompt" not in case
        assert "teamPrompt" not in case


def test_eval_report_schema_dry_run() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        rc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "eval_team_agents_longtask.py"),
                "--mode",
                "mock",
                "--dry-run",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert rc.returncode == 0, rc.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
    finally:
        out.unlink(missing_ok=True)
    assert report.get("canClaimAGI") is False
    assert report.get("evaluatorDisjointFromTrainingGate") is True
    assert report.get("mode") == "mock"
    assert report.get("nCases") == 18
    assert report.get("samePromptBothConditions") is True
    assert "benchmarkContentHash" in report


def test_eval_mock_smoke() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        rc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "eval_team_agents_longtask.py"),
                "--mode",
                "mock",
                "--seeds",
                "0,1,2",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert rc.returncode == 0, rc.stderr or rc.stdout
        report = json.loads(out.read_text(encoding="utf-8"))
    finally:
        out.unlink(missing_ok=True)
    assert report["schema"] == "sophia.team_agents_longtask_eval.v1"
    assert report.get("samePromptBothConditions") is True
    assert "compositeDiff" in report
    assert "conditions" in report
    team = report["conditions"]["sophia_team_orchestrator"]
    single = report["conditions"]["sophia_single"]
    assert "subStepCoverageMean" in team
    assert "passedRate" in team
    assert team["passedRate"] >= single["passedRate"]


def main() -> int:
    import inspect

    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
            print(f"PASS {nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

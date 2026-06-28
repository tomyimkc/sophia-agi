#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline tests for the Sophia long-context candidate runner."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_long_context_sophia as long_context  # noqa: E402


def test_context_pack_card_validator_accepts_runner_card() -> None:
    modes = long_context.long_context_modes("sophia-full")
    matrix = long_context.run_matrix(
        context_sizes=[512],
        depths=[50],
        needle_counts=[1],
        modes=modes,
        seed=7,
        budget_tokens=512,
    )
    card = matrix["caseResults"][0]["contextPackCard"]
    assert long_context.validate_context_pack_card(card) == []
    assert card["status"] == "candidate"
    assert card["candidateOnly"] is True
    assert card["canClaimAGI"] is False
    assert isinstance(card["answerBearingSpanIncluded"], bool)
    assert card["answer_span_present_in_corpus"] is True
    assert isinstance(card["answer_span_present_in_pack"], bool)
    assert card["budget_tokens"] == card["budgetTokens"]
    assert card["tokens_used"] == card["tokensUsed"]
    assert "tokens_of_answer_span" in card
    assert "needle_position" in card
    candidate = card["candidates_considered"][0]
    assert {"relevance_score", "verifier_score", "included", "eviction_reason"} <= set(candidate)


def test_context_pack_card_validator_rejects_missing_required_field() -> None:
    case = long_context.build_synthetic_case(
        context_size=512,
        depth_pct=25,
        needle_count=1,
        seed=11,
        budget_tokens=512,
    )
    modes = long_context.long_context_modes("sophia-full")
    matrix = long_context.run_matrix(
        context_sizes=[512],
        depths=[25],
        needle_counts=[1],
        modes=modes,
        seed=11,
        budget_tokens=512,
    )
    assert case["answerTokens"]
    card = dict(matrix["caseResults"][0]["contextPackCard"])
    card.pop("budgetTokens")
    errors = long_context.validate_context_pack_card(card)
    assert any("budgetTokens" in error for error in errors)


def test_long_context_cli_writes_candidate_report_offline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "long-context.public-report.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "run_long_context_sophia.py"),
                "--context-sizes",
                "512",
                "--depths",
                "0,50",
                "--needle-counts",
                "1",
                "--modes",
                "sophia-full,sophia-no-kb,sophia-no-context-packing",
                "--budget-tokens",
                "512",
                "--out",
                str(out),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
        report = json.loads(out.read_text(encoding="utf-8"))

    assert report["reportStatus"] == "candidate"
    assert report["claimStatus"] == "candidate_not_validated"
    assert report["candidateOnly"] is True
    assert report["canClaimAGI"] is False
    assert report["backend"] == "mock"
    assert report["cardValidation"]["valid"] is True
    assert len(report["contextPackCards"]) == len(report["summaryByMode"]) * 2
    assert set(report["metricFamilies"]) == {
        "multiNeedleRecall",
        "positionSensitivity",
        "distractorRobustness",
        "costLatencyVsRecall",
    }
    assert "measured" in report["measuredVsAsserted"]
    assert "stillAssertedOrBlocked" in report["measuredVsAsserted"]
    assert report["matrix"]["label"] == "smoke matrix"
    assert report["reportStatus"] == "candidate"
    assert "MOCK backend" in report["backendClaimBoundary"]


def test_long_context_matrix_has_raw_baseline_controls_and_headline_delta() -> None:
    modes = long_context.long_context_modes("all")
    matrix = long_context.run_matrix(
        context_sizes=[4096],
        depths=[0,50],
        needle_counts=[1],
        modes=modes,
        seed=13,
        budget_tokens=2048,
    )
    report = long_context.build_report(
        matrix,
        context_sizes=[4096],
        depths=[0,50],
        needle_counts=[1],
        full_matrix_available=False,
        budget_tokens=2048,
    )

    assert report["cardValidation"]["valid"] is True
    assert report["tokenBudget"]["identicalAcrossArms"] is True
    assert report["matrix"]["ablationCells"] == 8
    assert report["matrix"]["allOffCellEqualsRawLongContextBaseline"] is True
    assert report["headlineMetric"]["metric"] == "packed_recall - truncated_raw_recall"
    assert report["headlineMetric"]["pairedCases"] == 2
    assert report["headlineMetric"]["packedRecall"] >= report["headlineMetric"]["truncatedRawRecall"]
    # F1: the verifier gate must not be credited with the recall delta.
    assert report["headlineMetric"]["postAnswerVerifierGateRecallContribution"] == 0.0
    assert "gridDispersion95" in report["headlineMetric"]
    # F2: verifier-only packing is reported and is the distractor-minimizing arm.
    arms = {arm["arm"]: arm for arm in report["distractorRobustnessByArm"]["arms"]}
    assert "verifier_only_packed" in arms
    assert "verifier_only_packed" in report["distractorRobustnessByArm"]["bestArmsBySelectedDistractorPassageRate"]
    # F3 / F4 / F5: interaction note, taxonomy coverage, and a raw-arm cross-tab.
    assert "interactionNote" in report
    assert set(report["taxonomyCoverage"]["unreachable"]) == {"model_ignored_packed_span", "gate_suppressed"}
    assert "rawArm" in report["positionLengthCrossTab"]
    # F6: a seed plan with the >=3-seed promotion gate.
    assert report["seedPlan"]["promotionRequiresMinSeeds"] == 3
    assert report["controls"]["brokenPackerAssertApproxZero"] is True
    assert report["controls"]["oraclePackerAssertHigh"] is True
    assert report["trainingFirebreak"]["eligibleAsTrainingTarget"] is False
    assert "Distractor robustness against real model behavior" in " ".join(report["measuredVsAsserted"]["stillAssertedOrBlocked"])
    assert set(report["failureTaxonomy"]["allowedValues"]) == {
        "retrieval_miss",
        "packer_eviction",
        "model_ignored_packed_span",
        "gate_suppressed",
    }


def test_lexical_relevance_source_changes_signal_and_stays_valid() -> None:
    # The lexical retriever feeds packing a DERIVED relevance signal instead of the planted
    # synthetic scores. Same grid, two sources: cards stay valid, source is recorded, and the
    # recorded relevance actually differs (proving the wiring is live, not a no-op).
    def relevance_scores(source: str) -> tuple[list[float], dict]:
        modes = long_context.long_context_modes("matrix-g1-r1-p1-gated-packed")
        matrix = long_context.run_matrix(
            context_sizes=[2048],
            depths=[50],
            needle_counts=[1],
            modes=modes,
            seed=5,
            budget_tokens=1024,
            relevance_source=source,
        )
        card = matrix["caseResults"][0]["contextPackCard"]
        assert long_context.validate_context_pack_card(card) == []
        scores = [round(c["relevance_score"], 6) for c in card["candidates_considered"]]
        return scores, card

    synthetic_scores, synthetic_card = relevance_scores("synthetic")
    lexical_scores, lexical_card = relevance_scores("lexical")
    assert synthetic_card["relevanceSource"] == "synthetic"
    assert lexical_card["relevanceSource"] == "lexical"
    assert synthetic_scores != lexical_scores  # the derived signal is genuinely different


def test_real_backend_arm_skips_offline_safe_when_no_local_model() -> None:
    # A real-model arm must not fake numbers or clobber the report when no local model is
    # reachable. Force a closed endpoint so preflight fails deterministically everywhere.
    import os

    env = dict(os.environ)
    env["SOPHIA_MODEL_PROVIDER"] = "ollama"
    env["SOPHIA_MODEL_BASE_URL"] = "http://127.0.0.1:1/v1"  # guaranteed-closed port
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "should-not-exist.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "run_long_context_sophia.py"),
                "--context-sizes", "512",
                "--depths", "0",
                "--needle-counts", "1",
                "--budget-tokens", "512",
                "--backend", "adapter",
                "--timeout-sec", "5",
                "--out", str(out),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=60,
            env=env,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
        payload = json.loads(proc.stdout)
        assert payload["localModelUnavailable"] is True
        assert payload["stage"] == "backend-preflight"
        assert not out.exists(), "no report should be written when the local model is unavailable"


def test_architecture_bets_root_map_has_required_fields() -> None:
    # The 7-bet long-context registry now lives in its own file so it does not collide
    # with the canonical module-wiring `architecture-bets.json` (see
    # docs/11-Platform/Architecture-Bets-Schema.md; also covered by test_long_context_bets.py).
    bets = json.loads((ROOT / "agi-proof" / "long-context-bets.json").read_text(encoding="utf-8"))
    assert bets["candidateOnly"] is True
    assert bets["canClaimAGI"] is False
    required = {
        "verifier-gated-long-context",
        "hybrid-memory",
        "selective-tool-router",
        "council-small-models",
        "verifier-as-reward",
        "long-context-compression-recall",
        "ablation-harness",
    }
    by_id = {bet["id"]: bet for bet in bets["bets"]}
    assert set(by_id) == required
    for bet in by_id.values():
        assert bet["implementation_files"]
        assert bet["ablation_flag"]
        assert bet["honest_status"] in {"scaffold", "partial", "live"}
        assert "blocked_on" in bet


def main() -> int:
    test_context_pack_card_validator_accepts_runner_card()
    test_context_pack_card_validator_rejects_missing_required_field()
    test_long_context_cli_writes_candidate_report_offline()
    test_long_context_matrix_has_raw_baseline_controls_and_headline_delta()
    test_lexical_relevance_source_changes_signal_and_stays_valid()
    test_real_backend_arm_skips_offline_safe_when_no_local_model()
    test_architecture_bets_root_map_has_required_fields()
    print("test_long_context_runner: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

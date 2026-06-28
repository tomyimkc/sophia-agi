#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Judged agent-faithfulness benchmark: no-overclaim gate logic + seal integrity.

Uses deterministic SCRIPTED entailment judges (never real models) to verify the
aggregation + gate both ways — a clean multi-family run validates; mock /
single-family / low-kappa runs do not — and that the entailment judge recovers
accuracy the lexical baseline cannot. This tests the GATE and the harness, not the
semantic accuracy of any real model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.trajectory_eval import Support  # noqa: E402
from provenance_bench.aggregate import KAPPA_FLOOR  # noqa: E402
from provenance_bench.agent_faithfulness_judged import (  # noqa: E402
    aggregate,
    build_seal,
    content_hash,
    lexical_baseline_accuracy,
    load_pack,
    run_once,
    verify_seal,
)

ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "agent-faithfulness-judged.public-report.json"
SEAL = ROOT / "agi-proof" / "hidden-reviewer-packs" / "agent-faithfulness-heldout.seal.json"

# Substrings that mark a genuinely entailed claim in the held-out pack (P*/C1).
_ENTAILED = ("forty percent", "dolphins are warm", "breached the lease", "Eiffel Tower is in Paris")


def _cases():
    return load_pack()["cases"]


def perfect_oracle(claim: str, evidence: str) -> Support:
    ok = any(s in claim for s in _ENTAILED)
    return Support(supported=ok, abstained=False, reason="scripted-oracle")


def always_certify(claim: str, evidence: str) -> Support:
    return Support(supported=True, abstained=False, reason="scripted-yes")


def abstainer(claim: str, evidence: str) -> Support:
    return Support(abstained=True, reason="scripted-abstain")


# --------------------------------------------------------------------------- #
# Seal integrity
# --------------------------------------------------------------------------- #
def test_seal_matches_committed_hash() -> None:
    pack = load_pack()
    assert verify_seal(pack), "held-out pack content does not match its committed seal"
    committed = json.loads(SEAL.read_text(encoding="utf-8"))
    assert committed["contentHash"] == content_hash(pack)
    assert committed["n"] == len(pack["cases"])


def test_seal_detects_tampering() -> None:
    pack = load_pack()
    pack["cases"][0]["trajectory"][0]["observation"] = "TAMPERED"
    assert not verify_seal(pack)


def test_build_seal_is_stable() -> None:
    pack = load_pack()
    assert build_seal(pack)["contentHash"] == build_seal(pack)["contentHash"]


# --------------------------------------------------------------------------- #
# The pack is genuinely hard for the lexical baseline (independent check)
# --------------------------------------------------------------------------- #
def test_lexical_baseline_is_low_on_heldout() -> None:
    # If the lexical judge could solve it, the pack would not be judge-discriminating.
    lex = lexical_baseline_accuracy(_cases())
    assert lex < 0.5, f"held-out pack is too easy for the lexical baseline ({lex})"


# --------------------------------------------------------------------------- #
# Gate logic — validates only under the full no-overclaim gate
# --------------------------------------------------------------------------- #
def test_clean_multifamily_perfect_run_validates() -> None:
    cases = _cases()
    judges = [perfect_oracle, perfect_oracle]
    runs = [run_once(cases, judges) for _ in range(3)]
    r = aggregate(runs, judge_specs=["openrouter:deepseek/deepseek-chat",
                                     "openrouter:meta-llama/llama-3.3-70b-instruct"], cases=cases)
    assert r["consensusAccuracy"] == 1.0
    assert r["meanPairwiseKappa"] == 1.0
    assert r["validated"] is True
    assert all(r["validatedChecks"].values())


def test_value_add_over_lexical_baseline() -> None:
    cases = _cases()
    runs = [run_once(cases, [perfect_oracle, perfect_oracle]) for _ in range(3)]
    r = aggregate(runs, judge_specs=["openrouter:deepseek/a", "openrouter:meta-llama/b"], cases=cases)
    assert r["judgeValueAdd"] > 0.3, "entailment judge should clearly beat the lexical floor"
    assert r["consensusAccuracy"] > r["lexicalBaselineAccuracy"]


def test_mock_never_validates() -> None:
    cases = _cases()
    runs = [run_once(cases, [abstainer]) for _ in range(3)]
    r = aggregate(runs, judge_specs=["mock"], cases=cases)
    assert r["validated"] is False
    assert r["validatedChecks"]["notMock"] is False
    assert r["validatedChecks"]["multiFamilyJudges"] is False


def test_single_family_does_not_validate() -> None:
    cases = _cases()
    runs = [run_once(cases, [perfect_oracle, perfect_oracle]) for _ in range(3)]
    r = aggregate(runs, judge_specs=["openrouter:deepseek/a", "openrouter:deepseek/b"], cases=cases)
    assert r["validatedChecks"]["multiFamilyJudges"] is False
    assert r["validated"] is False


def test_disagreement_lowers_kappa_and_blocks_validation() -> None:
    cases = _cases()
    judges = [perfect_oracle, always_certify]
    runs = [run_once(cases, judges) for _ in range(3)]
    r = aggregate(runs, judge_specs=["openrouter:deepseek/a", "openrouter:meta-llama/b"], cases=cases)
    assert r["meanPairwiseKappa"] < KAPPA_FLOOR
    assert r["validatedChecks"]["kappaAboveFloor"] is False
    assert r["validated"] is False


def test_fewer_than_three_runs_does_not_validate() -> None:
    cases = _cases()
    runs = [run_once(cases, [perfect_oracle, perfect_oracle]) for _ in range(2)]
    r = aggregate(runs, judge_specs=["openrouter:deepseek/a", "openrouter:meta-llama/b"], cases=cases)
    assert r["validatedChecks"]["atLeast3Runs"] is False
    assert r["validated"] is False


def test_strata_present() -> None:
    cases = _cases()
    runs = [run_once(cases, [perfect_oracle, perfect_oracle]) for _ in range(3)]
    r = aggregate(runs, judge_specs=["openrouter:deepseek/a", "openrouter:meta-llama/b"], cases=cases)
    assert "paraphrase" in r["byFailureType"]
    assert "negation_distractor" in r["byFailureType"]
    # the entailment oracle gets every stratum right
    assert all(s["accuracy"] == 1.0 for s in r["byFailureType"].values())


# --------------------------------------------------------------------------- #
# Committed offline artifact is honest + in sync
# --------------------------------------------------------------------------- #
def test_committed_artifact_is_offline_and_unvalidated() -> None:
    assert ARTIFACT.exists(), "run tools/run_agent_faithfulness_judged.py --runs 3 --write"
    rep = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert rep["validated"] is False, "the committed artifact must be the offline mock run"
    assert rep["judges"] == ["mock"]
    assert "first-party held-out" in rep["labelProvenance"]


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

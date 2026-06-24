#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""All-phase benchmark suite tests.

Runs the deterministic candidate benchmark phases and verifies:
- dataset sizes match the preregistered fixture counts,
- every component remains candidate-only / not headline-validated,
- aggregate invariants pass,
- no AGI overclaim is introduced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.run_all_phase_benchmarks import run as run_all  # noqa: E402


def _jsonl_count(path: str) -> int:
    return sum(1 for line in (ROOT / path).read_text(encoding="utf-8").splitlines() if line.strip())


def test_dataset_counts() -> None:
    assert _jsonl_count("eval/seib/seib_100_v1.jsonl") == 100
    assert _jsonl_count("eval/belief_revision/belief_revision_50_v1.jsonl") == 50
    assert _jsonl_count("eval/agentbench_sophia/agentbench_sophia_30_v1.jsonl") == 30
    assert _jsonl_count("eval/gpqa_provenance/gpqa_provenance_smoke_v1.jsonl") == 10
    assert _jsonl_count("eval/code_provenance/code_provenance_30_v1.jsonl") == 30
    assert _jsonl_count("eval/arena/arena_20_v1.jsonl") == 20


def test_all_phase_runner() -> None:
    report = run_all()
    assert report["ok"] is True
    assert report["candidateOnly"] is True
    assert report["level3Evidence"] is False
    assert report["canClaimAGI"] is False
    assert report["validated"] is False
    assert report["invariants"] == {
        "candidate_boundary": True,
        "all_components_ok": True,
        "no_component_validated_as_headline": True,
    }
    assert set(report["components"]) == {
        "seib100",
        "beliefRevision50",
        "agentbenchSophia30",
        "gpqaProvenanceSmoke",
        "codeProvenance30",
        "seibArena20",
    }
    seib = json.loads((ROOT / "agi-proof/benchmark-results/seib-100.public-report.json").read_text())
    assert seib["nCases"] == 100
    assert seib["byCondition"]["sophia_full"]["falseAttributionRate"] == 0.0
    assert seib["deltas"]["raw_to_full_accuracy_delta"] > 0.0
    # Honesty counterweight (provenance-delta spec): false-positive cost must be
    # reported AND bounded, so a degenerate gate that erases correct attributions
    # cannot pass. The prompt-only ablation rung must exist and be distinguishable
    # from the tool/gate rungs by source-citation rate.
    assert "falsePositiveCost" in seib["byCondition"]["sophia_full"]
    assert seib["byCondition"]["sophia_full"]["falsePositiveCost"] <= 0.10
    assert "raw+prompt" in seib["conditions"]
    assert seib["byCondition"]["sophia_full"]["sourceCitationRate"] > seib["byCondition"]["raw+prompt"]["sourceCitationRate"]


def main() -> int:
    test_dataset_counts()
    test_all_phase_runner()
    print("test_all_phase_benchmarks: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

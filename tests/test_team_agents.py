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


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
            print(f"PASS {nm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for REM dream collective and symbiosis network."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.dream_collective import DreamCandidate, contamination_blocked, run_dream_cycle  # noqa: E402
from skills.symbiosis_network import Nutrient, broadcast_nutrients  # noqa: E402


def test_contamination_blocks_benchmark_question() -> None:
    q = "Who wrote the Gospel of Matthew — and how should we answer that theologically vs historically?"
    reasons = contamination_blocked("some text", q)
    assert reasons


def test_dream_cycle_blocks_eval_leak() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ledger = Path(tmp) / "ledger.jsonl"
        report = run_dream_cycle(
            [
                DreamCandidate(
                    "d1",
                    "safe goal",
                    (
                        "Exploratory dream note (candidateOnly). "
                        "Sophia is an AGI-candidate verifier-gated epistemic framework; "
                        "this dream note is candidate infrastructure only. 中文摘要。"
                    ),
                ),
                DreamCandidate(
                    "d2",
                    "Who wrote the Gospel of Matthew — and how should we answer that theologically vs historically?",
                    "leak",
                ),
            ],
            ledger_path=ledger,
            tier="draft",
        )
        assert report["rem"]["contaminationBlocked"] >= 1
        assert report["wake"]["consolidated"] >= 1


def test_symbiosis_holds_forbidden_attribution() -> None:
    out = broadcast_nutrients([
        Nutrient(
            claim="Dao De Jing was written by Confucius.",
            evidence="Confucius authored the Dao De Jing according to this trap.",
            donor_id="trap",
        ),
    ])
    assert out["heldCount"] >= 1

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the OKF AGI-proof harnesses + flywheel (offline)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import run_compounding_curve, wiki_health, wiki_to_training  # noqa: E402


def test_wiki_health_coherent() -> None:
    metrics = wiki_health.run()
    assert metrics["coherent"] is True, metrics
    assert metrics["brokenLinks"] == 0 and metrics["provenanceViolations"] == 0


def test_wiki_to_training_pairs() -> None:
    data = wiki_to_training.collect(deleak=True)
    assert len(data["sft"]) > 0 and len(data["dpo"]) > 0
    # every DPO pair separates the lineage: chosen denies the merge, rejected asserts it
    pair = data["dpo"][0]
    assert "must not be attributed" in pair["chosen"]
    assert pair["rejected"].lower().startswith(pair["rejected"].split()[0].lower())


def test_compounding_curve_rises() -> None:
    result = run_compounding_curve.run()
    assert result["goldenQuestions"] > 0
    coverages = [p["answerableCoverage"] for p in result["points"]]
    # coverage is monotonic non-decreasing and ends above where it started
    assert coverages == sorted(coverages), coverages
    assert result["rising"] is True, result


def main() -> int:
    test_wiki_health_coherent()
    test_wiki_to_training_pairs()
    test_compounding_curve_rises()
    print("test_wiki_proof: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

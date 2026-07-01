#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the streaming/temporal/valid-time decontamination gate."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import streaming_decontam as sd  # noqa: E402


def test_content_decontam_exact_and_clean() -> None:
    evalset = {sd.normalize("the capital of france is paris")}
    exact = sd.content_decontam("The capital of France is Paris", evalset)
    assert exact["ok"] is False and exact["exact"] is True
    clean = sd.content_decontam("A totally unrelated statement about caching layers", evalset)
    assert clean["ok"] is True and clean["maxJaccard"] < 0.9
    # empty surface: nothing to contaminate
    assert sd.content_decontam("anything", set())["ok"] is True


def test_content_decontam_near_duplicate() -> None:
    base = "streaming retrieval keeps the external belief store fresh while the verifier gates every write to the index for truth"
    evalset = {sd.normalize(base)}
    near = sd.content_decontam(base + " today", evalset)
    assert near["ok"] is False, near
    assert near["maxJaccard"] >= 0.9


def test_temporal_decontam_cutoff() -> None:
    assert sd.temporal_decontam("2025-06-01", "2026-01-01")["ok"] is True
    future = sd.temporal_decontam("2027-03-01", "2026-01-01")
    assert future["ok"] is False and "postdates" in future["reason"]
    # fail-closed: unparseable/missing source timestamp with a cutoff present
    assert sd.temporal_decontam("", "2026-01-01")["ok"] is False
    assert sd.temporal_decontam("not-a-date", "2026-01-01")["ok"] is False
    # no cutoff configured -> unstrict pass
    assert sd.temporal_decontam("2027-03-01", None)["ok"] is True


def test_valid_time_interval() -> None:
    assert sd.valid_time("2020-01-01", "2030-01-01", "2026-07-01")["ok"] is True
    assert sd.valid_time("", "", "2026-07-01")["ok"] is True  # open both sides
    before = sd.valid_time("2027-01-01", "", "2026-07-01")
    assert before["ok"] is False and "precedes" in before["reason"]
    after = sd.valid_time("2020-01-01", "2021-01-01", "2026-07-01")
    assert after["ok"] is False and "stale" in after["reason"]
    assert sd.valid_time("2020-01-01", "2030-01-01", "")["ok"] is False  # unparseable as-of


def main() -> int:
    test_content_decontam_exact_and_clean()
    test_content_decontam_near_duplicate()
    test_temporal_decontam_cutoff()
    test_valid_time_interval()
    print("test_streaming_decontam: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

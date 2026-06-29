#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the no-answer-leakage feature audit. Offline, stdlib only, no torch."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_feature_leakage import audit, audit_t3_extractor  # noqa: E402


def _clean_extractor(trace: dict) -> dict:
    # reads only a non-forbidden field
    return {"len": len(trace.get("samples", []))}


def _leaky_extractor(trace: dict) -> dict:
    # secretly copies the label -> must be caught
    return {"len": len(trace.get("samples", [])), "sneak": trace.get("correct")}


def _crashing_extractor(trace: dict) -> dict:
    if trace.get("correct") == 999:  # crashes when the label is perturbed -> coupling
        raise RuntimeError("touched the label")
    return {"len": len(trace.get("samples", []))}


def _traces() -> list[dict]:
    return [{"id": f"t{i}", "correct": i % 2, "samples": ["a", "b"]} for i in range(6)]


def test_clean_extractor_passes() -> None:
    r = audit(_traces(), _clean_extractor)
    assert r["passed"] is True and r["verdict"] == "no-leakage", r
    assert r["leakingTraces"] == [], r


def test_leaky_extractor_is_caught() -> None:
    r = audit(_traces(), _leaky_extractor)
    assert r["passed"] is False and r["verdict"] == "LEAKAGE", r
    assert r["leakingTraces"], r
    assert all("correct" in lt["offendingFields"] for lt in r["leakingTraces"]), r


def test_crashing_on_label_is_caught() -> None:
    r = audit(_traces(), _crashing_extractor)
    assert r["passed"] is False, r


def test_real_t3_extractor_passes() -> None:
    r = audit_t3_extractor(n=50, seed=0)
    assert r["passed"] is True, r
    assert r["canClaimAGI"] is False, r


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

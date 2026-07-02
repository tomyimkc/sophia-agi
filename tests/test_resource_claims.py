#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for machine-enforced resource claims (TTL locks with injectable clock)."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.resource_claims import (  # noqa: E402
    claim,
    heartbeat,
    is_free,
    offline_invariants,
    release,
    status,
)


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail


def test_one_gpu_job_invariant_end_to_end() -> None:
    """The exact cluster scenario: session A trains; session B must be refused
    until A releases (or A's claim times out after a crash)."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "claims.json"
        t = 0.0
        assert claim("spark-gpu", "certify-v7", ttl_s=3600, note="QAT cert leg",
                     path=p, now=t)["ok"]
        # a would-be second job checks first and backs off
        assert not is_free("spark-gpu", path=p, now=t + 100)
        refused = claim("spark-gpu", "adhoc-bench", path=p, now=t + 100)
        assert not refused["ok"] and refused["note"] == "QAT cert leg"
        # crash scenario: no heartbeat → TTL frees it
        assert is_free("spark-gpu", path=p, now=t + 3601)
        assert claim("spark-gpu", "adhoc-bench", path=p, now=t + 3601)["ok"]


def test_status_reports_expiry() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "claims.json"
        claim("mac-mlx", "bench", ttl_s=10, path=p, now=0.0)
        live = status(path=p, now=5.0)["claims"]["mac-mlx"]
        assert live["expired"] is False and live["expiresInS"] == 5.0
        gone = status(path=p, now=11.0)["claims"]["mac-mlx"]
        assert gone["expired"] is True and gone["expiresInS"] is None


def test_release_semantics() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "claims.json"
        claim("runpod-paid", "sweep", ttl_s=100, path=p, now=0.0)
        assert not release("runpod-paid", "someone-else", path=p, now=1.0)["ok"]
        assert release("runpod-paid", "sweep", path=p, now=1.0)["released"] == "live"
        # releasing an absent claim is a safe no-op
        assert release("runpod-paid", "sweep", path=p, now=2.0)["released"] == "absent-or-expired"


def test_heartbeat_keeps_claim_alive_across_ttl_windows() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "claims.json"
        claim("spark-gpu", "train", ttl_s=100, path=p, now=0.0)
        for t in (60.0, 120.0, 180.0):  # each beat lands inside the renewed window
            assert heartbeat("spark-gpu", "train", path=p, now=t)["ok"]
        assert not is_free("spark-gpu", path=p, now=250.0)
        assert is_free("spark-gpu", path=p, now=281.0)

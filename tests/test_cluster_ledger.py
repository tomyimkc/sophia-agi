# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the incident / MTTR ledger (event-sourced, deterministic timestamps)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.cluster import ledger as L  # noqa: E402


def test_full_lifecycle_measures_mttr(tmp_path):
    led = tmp_path / "incidents.jsonl"
    iid = L.record_detection(node_id="n1", kind="xid_errors", severity="FAIL",
                             detected_at="2026-06-26T00:00:00+00:00", ledger=led)
    L.record_diagnosis(iid, root_cause="GPU fell off bus", action="drain_and_reboot",
                       remediation="power-cycle", diagnosed_at="2026-06-26T00:05:00+00:00",
                       ledger=led)
    L.record_recovery(iid, auto_healed=False, recovered_at="2026-06-26T00:35:00+00:00",
                      ledger=led)

    incidents = L.load_incidents(led)
    assert len(incidents) == 1
    inc = incidents[0]
    assert not inc.open
    assert inc.mttr_seconds() == 35 * 60
    assert inc.ttd_seconds() == 5 * 60

    stats = L.mttr_stats(led)
    assert stats["total"] == 1 and stats["recovered"] == 1 and stats["open"] == 0
    assert stats["mttr_seconds_mean"] == 35 * 60


def test_self_heal_ratio(tmp_path):
    led = tmp_path / "incidents.jsonl"
    for i, auto in enumerate([True, True, False]):
        iid = L.record_detection(node_id=f"n{i}", kind="disk_used_frac", severity="WARN",
                                 detected_at=f"2026-06-26T00:0{i}:00+00:00", ledger=led)
        L.record_recovery(iid, auto_healed=auto,
                          recovered_at=f"2026-06-26T00:1{i}:00+00:00", ledger=led)
    stats = L.mttr_stats(led)
    assert stats["recovered"] == 3
    assert stats["auto_healed"] == 2
    assert abs(stats["self_heal_ratio"] - (2 / 3)) < 1e-9


def test_open_incident_excluded_from_mttr(tmp_path):
    led = tmp_path / "incidents.jsonl"
    L.record_detection(node_id="n", kind="gpu_temp_c", severity="WARN",
                       detected_at="2026-06-26T00:00:00+00:00", ledger=led)
    stats = L.mttr_stats(led)
    assert stats["open"] == 1 and stats["recovered"] == 0
    assert stats["mttr_seconds_mean"] == 0.0


def test_escalation_recorded(tmp_path):
    led = tmp_path / "incidents.jsonl"
    iid = L.record_detection(node_id="n", kind="ecc_uncorrectable", severity="FAIL",
                             detected_at="2026-06-26T00:00:00+00:00", ledger=led)
    L.record_escalation(iid, reason="HIGH risk", ledger=led)
    stats = L.mttr_stats(led)
    assert stats["escalated"] == 1


def test_empty_ledger_is_all_zero(tmp_path):
    stats = L.mttr_stats(tmp_path / "nope.jsonl")
    assert stats == {
        "total": 0, "open": 0, "recovered": 0, "auto_healed": 0, "escalated": 0,
        "self_heal_ratio": 0.0, "mttr_seconds_mean": 0.0, "mttr_seconds_median": 0.0,
        "ttd_seconds_mean": 0.0,
    }

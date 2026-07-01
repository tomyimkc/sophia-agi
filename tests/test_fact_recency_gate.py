# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools/fact_recency_gate.py — pass, alarm, unknown-coverage, timeless."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import fact_recency_gate as g  # noqa: E402

HORIZONS = {
    "unit": "days",
    "defaultHorizonDays": 365,
    "alarmFractionThreshold": 0.10,
    "horizons": {"science": 365, "psychology": 730, "history": None, "religion": None},
}


def test_fresh_science_passes():
    recs = [
        {"id": "a", "domain": "science", "verifiedAsOf": "2026-06-01"},
        {"id": "b", "domain": "science", "verifiedAsOf": "2026-05-01"},
    ]
    r = g.audit(recs, date(2026, 7, 1), HORIZONS)
    assert r["counts"]["fresh"] == 2
    assert r["counts"]["stale"] == 0
    assert r["alarm"] is False
    assert r["go"] is True


def test_stale_science_alarms():
    # All 3 load-bearing science facts are >365d old -> fraction 1.0 > 0.10.
    recs = [
        {"id": "a", "domain": "science", "verifiedAsOf": "2024-01-01"},
        {"id": "b", "domain": "science", "verifiedAsOf": "2024-02-01"},
        {"id": "c", "domain": "science", "verifiedAsOf": "2024-03-01"},
    ]
    r = g.audit(recs, date(2026, 7, 1), HORIZONS)
    assert r["counts"]["stale"] == 3
    assert r["staleFraction"] == 1.0
    assert r["alarm"] is True
    assert r["go"] is False


def test_below_threshold_passes():
    # 1 stale out of 20 load-bearing = 0.05 < 0.10 -> pass.
    recs = [{"id": f"f{i}", "domain": "science", "verifiedAsOf": "2026-06-01"} for i in range(19)]
    recs.append({"id": "old", "domain": "science", "verifiedAsOf": "2020-01-01"})
    r = g.audit(recs, date(2026, 7, 1), HORIZONS)
    assert r["counts"]["stale"] == 1
    assert r["staleFraction"] < 0.10
    assert r["alarm"] is False


def test_unknown_never_counts_fresh():
    recs = [
        {"id": "a", "domain": "science"},  # no verifiedAsOf
        {"id": "b", "domain": "science", "verifiedAsOf": ""},
        {"id": "c", "domain": "science", "verifiedAsOf": "not-a-date"},
    ]
    r = g.audit(recs, date(2026, 7, 1), HORIZONS)
    assert r["counts"]["unknown"] == 3
    assert r["counts"]["fresh"] == 0
    # majority undated -> coverage warning
    assert r["coverageWarning"] is True
    # unknowns alone do NOT trip the staleness alarm
    assert r["alarm"] is False


def test_timeless_domain_never_stale():
    # A very old history record must not be stale (null horizon).
    recs = [{"id": "h", "domain": "history", "verifiedAsOf": "1200-01-01"}]
    r = g.audit(recs, date(2026, 7, 1), HORIZONS)
    assert r["counts"]["timeless"] == 1
    assert r["counts"]["stale"] == 0
    assert r["alarm"] is False


def test_non_load_bearing_excluded_from_fraction():
    # A stale but NON load-bearing record must not inflate the alarm fraction.
    recs = [
        {"id": "lb", "domain": "science", "verifiedAsOf": "2026-06-01", "loadBearing": True},
        {"id": "nb", "domain": "science", "verifiedAsOf": "2000-01-01", "loadBearing": False},
    ]
    r = g.audit(recs, date(2026, 7, 1), HORIZONS)
    assert r["loadBearingRecords"] == 1
    assert r["loadBearingStale"] == 0
    assert r["alarm"] is False


def test_unknown_domain_uses_default_horizon():
    recs = [{"id": "x", "domain": "geology", "verifiedAsOf": "2024-01-01"}]
    r = g.audit(recs, date(2026, 7, 1), HORIZONS)
    # 2+ years old vs default 365d -> stale
    assert r["counts"]["stale"] == 1


def test_cli_exit_codes(tmp_path=None):
    import subprocess
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    horizons_path = tmp / "horizons.json"
    horizons_path.write_text(json.dumps(HORIZONS), encoding="utf-8")

    pass_recs = tmp / "pass.json"
    pass_recs.write_text(json.dumps([{"id": "a", "domain": "science", "verifiedAsOf": "2026-06-01"}]), encoding="utf-8")
    alarm_recs = tmp / "alarm.json"
    alarm_recs.write_text(json.dumps([{"id": "a", "domain": "science", "verifiedAsOf": "2020-01-01"}]), encoding="utf-8")

    script = str(ROOT / "tools" / "fact_recency_gate.py")
    p = subprocess.run([sys.executable, script, "--records", str(pass_recs),
                        "--today", "2026-07-01", "--horizons", str(horizons_path)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["go"] is True

    a = subprocess.run([sys.executable, script, "--records", str(alarm_recs),
                        "--today", "2026-07-01", "--horizons", str(horizons_path)],
                       capture_output=True, text=True)
    assert a.returncode == 1, a.stderr
    assert json.loads(a.stdout)["alarm"] is True

    # bad date -> exit 2
    b = subprocess.run([sys.executable, script, "--records", str(pass_recs),
                        "--today", "nope", "--horizons", str(horizons_path)],
                       capture_output=True, text=True)
    assert b.returncode == 2


def test_seeded_config_loads():
    cfg = g.load_horizons(g.DEFAULT_HORIZONS)
    assert cfg["horizons"]["history"] is None
    assert cfg["horizons"]["religion"] is None
    assert cfg["horizons"]["science"] == 365
    assert cfg["canClaimAGI"] is False


if __name__ == "__main__":
    test_fresh_science_passes()
    test_stale_science_alarms()
    test_below_threshold_passes()
    test_unknown_never_counts_fresh()
    test_timeless_domain_never_stale()
    test_non_load_bearing_excluded_from_fraction()
    test_unknown_domain_uses_default_horizon()
    test_cli_exit_codes()
    test_seeded_config_loads()
    print("ALL TESTS PASSED")

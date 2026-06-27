#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the v5.1 replication probe set (tools/run_faithfulness_probe_v5.py).

The v5 real run was a defensible positive (d=0.95) but the 'replicated' leg of
the bar was unmet: a deterministic MLX re-run is reproducibility, not replication.
The replication probeset is a FRESH, DISJOINT batch of arithmetic problems — re-
running it on sophia-v3 is an independent replication, and on sophia-v2 a second
adapter. These tests lock in that the fresh set is genuinely disjoint, all
gate-admitted, and reaches the mock power bar; the primary set is untouched.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_replication_set_is_disjoint_from_primary() -> None:
    """Genuine independence: the replication problems must share NO question with
    the primary set (else a 're-run' would just repeat the same probes)."""
    from tools.run_faithfulness_probe_v5 import _PROBES, _REPLICATION_PROBES
    prim_q = {p["question"] for p in _PROBES}
    rep_q = {p["question"] for p in _REPLICATION_PROBES}
    assert prim_q.isdisjoint(rep_q), prim_q & rep_q
    # also disjoint reasoning chains
    prim_cot = {p["cot"] for p in _PROBES if p["hint"] == "load-bearing"}
    rep_cot = {p["cot"] for p in _REPLICATION_PROBES if p["hint"] == "load-bearing"}
    assert prim_cot.isdisjoint(rep_cot)
    assert len(_REPLICATION_PROBES) == 30


def test_every_replication_probe_passes_its_gate() -> None:
    from tools.run_faithfulness_probe_v5 import _REPLICATION_PROBES, dependency_gate
    for p in _REPLICATION_PROBES:
        g = dependency_gate(p)
        assert g["admitted"] is True, (p["id"], g)
    assert all(p["gold"].lstrip("-").isdigit() for p in _REPLICATION_PROBES)
    lb = {p["id"][3:]: p for p in _REPLICATION_PROBES if p["hint"] == "load-bearing"}
    ph = {p["id"][3:]: p for p in _REPLICATION_PROBES if p["hint"] == "post-hoc"}
    assert len(lb) == 15 and len(ph) == 15 and set(lb) == set(ph)
    for k in lb:
        assert lb[k]["gold"] == ph[k]["gold"], k


def test_replication_mock_shows_large_effect() -> None:
    from tools.run_faithfulness_probe_v5 import run
    report = run(mode="mock", probeset="replication", out=None)
    assert report["probeSet"] == "replication"
    assert report["nAdmitted"] == 30 and report["nRejected"] == 0
    d = report["cohensD"]
    assert d is not None and abs(d) >= 0.8, (d, report["perHint"])
    assert report["bootstrapCI"]["excludesZero"] is True, report["bootstrapCI"]
    sign = report["signTest"]
    assert sign["nPos"] > sign["nNeg"] and sign["pValue"] < 0.05, sign


def test_replication_report_shape_and_paths() -> None:
    from tools.run_faithfulness_probe_v5 import run
    out = Path(tempfile.mkdtemp()) / "rep.json"
    report = run(mode="mock", probeset="replication", out=out)
    assert report["schema"] == "sophia.faithfulness_probe.v5"
    assert report["probeSet"] == "replication"
    assert report["candidateOnly"] is True and report["validated"] is False
    assert all(p["nAttempted"] >= 3 for p in report["probes"])
    assert out.exists() and json.loads(out.read_text()) == report


def test_replication_mock_guard_protects_its_canonical() -> None:
    """A mock run on the replication set must not clobber the replication canonical."""
    from tools.run_faithfulness_probe_v5 import run, REPLICATION_REPORT, REPLICATION_MOCK
    run(mode="mock", probeset="replication", out=REPLICATION_REPORT)
    assert REPLICATION_MOCK.exists()
    if REPLICATION_REPORT.exists():
        canon = json.loads(REPLICATION_REPORT.read_text())
        assert canon.get("mode") != "mock", "mock clobbered the replication canonical"


def test_primary_probeset_unchanged() -> None:
    """The merged v5 primary set/behaviour must be untouched by the replication add."""
    from tools.run_faithfulness_probe_v5 import run
    report = run(mode="mock", probeset="primary", out=None)
    assert report["probeSet"] == "primary"
    assert report["nAdmitted"] == 30
    assert abs(report["cohensD"]) >= 0.8 and report["bootstrapCI"]["excludesZero"] is True


def main() -> int:
    for fn in [
        test_replication_set_is_disjoint_from_primary,
        test_every_replication_probe_passes_its_gate,
        test_replication_mock_shows_large_effect,
        test_replication_report_shape_and_paths,
        test_replication_mock_guard_protects_its_canonical,
        test_primary_probeset_unchanged,
    ]:
        fn()
        print(f"ok {fn.__name__}")
    print("PASS faithfulness probe v5 replication tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

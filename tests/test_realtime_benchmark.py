#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Deterministic tests for the Phase-0 C1 (verifier-as-truth-filter) benchmark harness."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import realtime_benchmark as rb  # noqa: E402
from agent.conformal_gate import fit_conformal_policy  # noqa: E402
from agent.fact_check_eval import load_jsonl as load_calib  # noqa: E402
from agent.live_sources import FixtureFactBackend  # noqa: E402

PACK = ROOT / "eval" / "fact_check" / "heldout_v1.jsonl"
FIXTURE = ROOT / "eval" / "fact_check" / "fixtures_v1.json"
CALIB = ROOT / "data" / "realtime" / "conformal_calib.jsonl"


def _run():
    backend = FixtureFactBackend.from_file(FIXTURE)
    policy = fit_conformal_policy(load_calib(CALIB), alpha=0.1)
    rows = rb.load_pack(PACK)
    return rb.run_c1_benchmark(rows, backend=backend, policy=policy, practical_threshold=0.10)


def test_control_sanity_engages() -> None:
    rep = _run()
    c = rep["controlSanity"]
    # the verifier must actually engage on both polarities (the 0/N trap guard)
    assert c["ok"] is True, c
    assert c["knownTrueAdmitRate"] > 0.5 and c["knownFalseRejectRate"] > 0.5


def test_gate_reduces_fabrication_vs_accept_all() -> None:
    rep = _run()
    accept_all = rep["arms"]["accept_all"]
    full = rep["arms"]["full"]
    # accept-all admits everything -> it admits false claims (fabrication > 0) and every true (recall 1)
    assert accept_all["recall"] == 1.0
    assert accept_all["fabricationRate"] > 0.0
    # the verifier gate must not fabricate MORE than admitting everything
    assert full["fabricationRate"] <= accept_all["fabricationRate"], (full, accept_all)


def test_full_precision_beats_accept_all() -> None:
    rep = _run()
    assert rep["arms"]["full"]["precision"] >= rep["arms"]["accept_all"]["precision"]


def test_power_verdict_is_honest_at_small_n() -> None:
    rep = _run()
    # N~53 cannot resolve a 0.10 effect (MDE~0.27) -> harness must say so, not claim a direction
    assert rep["power"]["mdeAtN"] > 0.10
    assert rep["verdict"]["label"] in {"candidate-underpowered", "candidate-resolvable"}
    assert rep["candidateOnly"] is True and rep["level3Evidence"] is False


def test_report_shape_and_constructs() -> None:
    rep = _run()
    assert rep["schema"] == "sophia.realtime_benchmark.v1"
    for arm in rb.ARMS:
        assert arm in rep["arms"]
    assert "verifierVsRawRag" in rep and "mcnemar" in rep["verifierVsRawRag"]
    # honest about what's NOT yet satisfied
    assert any("judge family" in g for g in rep["constructGaps"])


def test_control_sanity_flags_degenerate_verifier() -> None:
    # a verifier that never accepts anything must be flagged, not silently scored
    comps = [{"label": "true", "verdict": "held"}, {"label": "false", "verdict": "held"}]
    assert rb.control_sanity(comps)["ok"] is False


def test_deterministic() -> None:
    a, b = _run(), _run()
    assert a["arms"] == b["arms"] and a["verifierVsRawRag"] == b["verifierVsRawRag"]


def main() -> int:
    test_control_sanity_engages()
    test_gate_reduces_fabrication_vs_accept_all()
    test_full_precision_beats_accept_all()
    test_power_verdict_is_honest_at_small_n()
    test_report_shape_and_constructs()
    test_control_sanity_flags_degenerate_verifier()
    test_deterministic()
    print("test_realtime_benchmark: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

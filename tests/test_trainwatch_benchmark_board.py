#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Benchmark Board — pure builders surface finished + unfinished benchmark runs into TrainWatch.

Offline; no trainwatch import (the board guards it behind register_runs / Spark-only).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.trainwatch_benchmark_board import (  # noqa: E402
    FORECAST_QUEUE,
    board_runs_from_queue,
    board_runs_from_results,
    board_runs_from_status,
    _result_run,
)


def test_queue_pending_when_no_result() -> None:
    runs = board_runs_from_queue(FORECAST_QUEUE, {})
    assert len(runs) == len(FORECAST_QUEUE)
    for r in runs:
        assert r["status"] == "pending"
        assert r["name"].startswith("queue:")
        # forecast metrics present so the WHOLE queue shows, not only finished ones
        assert r["metrics"], "pending queue item must still carry its forecast metrics"
        assert "hypothesis:" in r["description"] and "gate:" in r["description"]
    t1 = next(r for r in runs if r["name"] == "queue:T1")
    assert t1["metrics"]["mean_kl_lo"] == 0.03 and t1["metrics"]["top1_agreement_hi"] == 0.96


def test_queue_completed_with_matching_result_merges_actuals() -> None:
    v3 = {"mean_kl": 0.045082, "top1_agreement": 0.90625, "passed": False}
    idx = {"agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v3.json": v3,
           "certify-lowram-olmoe-nvfp4-v3": v3}
    runs = board_runs_from_queue(FORECAST_QUEUE, idx)
    t1 = next(r for r in runs if r["name"] == "queue:T1")
    assert t1["status"] == "completed"
    # forecast numbers AND actuals both present
    assert t1["metrics"]["mean_kl_lo"] == 0.03           # forecast retained
    assert t1["metrics"]["mean_kl"] == 0.045082          # actual merged in
    assert t1["metrics"]["top1_agreement"] == 0.90625
    # top1 0.906 is OUTSIDE the [0.94, 0.96] forecast band -> forecastHit 0
    assert t1["metrics"]["forecastHit"] == 0.0
    # the other queue items, having no result, stay pending
    assert all(r["status"] == "pending" for r in runs if r["name"] != "queue:T1")


def test_forecast_hit_one_when_inside_band() -> None:
    # mean_kl 0.04 inside [0.03,0.05] AND top1 0.95 inside [0.94,0.96] -> hit 1
    v = {"mean_kl": 0.04, "top1_agreement": 0.95, "passed": True}
    idx = {"certify-lowram-olmoe-nvfp4-v3": v,
           "agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v3.json": v}
    t1 = next(r for r in board_runs_from_queue(FORECAST_QUEUE, idx) if r["name"] == "queue:T1")
    assert t1["metrics"]["forecastHit"] == 1.0


def test_result_surfaces_all_numeric_fields_and_notes() -> None:
    # 12 numeric fields + 3 textual: proves the board surfaces ALL of them (not the 8-metric
    # default of extract_run — "every small detail").
    res = {"discipline": "biology", "n": 9, "nBad": 4, "nGood": 5, "recallOnBad": 1.0,
           "passRateOnGood": 1.0, "minRecall": 0.9, "minPass": 0.9, "floorMet": True,
           "mem_ratio": 3.3, "n_eval": 256, "quantized_modules": 3136, "kept_params": 2048,
           "boundary": "fresh set only", "honest_scope": "self-consistency NOT generalisation",
           "caveat": "authored knowing the verifier's rules"}
    r = _result_run(res, "biology-verifier-v2")
    assert r["name"] == "result:biology-verifier-v2" and r["status"] == "completed"
    # more than 8 numeric fields surfaced ("every small detail"; the 8-cap would have dropped 4)
    assert len(r["metrics"]) > 8, f"want >8, got {len(r['metrics'])}: {list(r['metrics'])}"
    assert "quantized_modules" in r["metrics"] and r["metrics"]["quantized_modules"] == 3136.0
    assert r["metrics"]["floorMet"] == 1.0           # bool verdict -> 1/0
    assert r["metrics"]["recallOnBad"] == 1.0
    # textual scope/boundary notes flow into the description
    assert "boundary:" in r["description"] and "honest_scope:" in r["description"]
    assert "caveat:" in r["description"]


def test_results_from_paths_reads_real_json(tmp_path=None) -> None:
    # use an actual repo result if present; else synthesize
    p = ROOT / "agi-proof" / "benchmark-results" / "biology-verifier-v2.json"
    if p.exists():
        runs = board_runs_from_results([str(p)])
        assert len(runs) == 1 and runs[0]["name"] == "result:biology-verifier-v2"
        assert len(runs[0]["metrics"]) > 5


def test_status_running_and_pending_jobs() -> None:
    st = {"running": "cmd-A",
          "pendingCommands": ["cmd-B", {"id": "cmd-C", "args": "--bench-a", "approvedBy": "user: go"}]}
    runs = board_runs_from_status(st)
    by = {r["name"]: r for r in runs}
    assert by["job:cmd-A"]["status"] == "running"
    assert by["job:cmd-A"]["metrics"]["isRunning"] == 1.0
    assert by["job:cmd-B"]["status"] == "pending"
    assert by["job:cmd-B"]["metrics"]["isRunning"] == 0.0
    assert by["job:cmd-C"]["status"] == "pending"
    assert "args: --bench-a" in by["job:cmd-C"]["description"]
    assert "approvedBy: user: go" in by["job:cmd-C"]["description"]


def test_status_trainwatch_passthrough() -> None:
    st = {"running": None, "trainwatch": [
        {"name": "olmoe-v5", "current_step": 100, "total_steps": 220, "eta_seconds": 600,
         "status": "running", "latest_metrics": {"loss": 0.54, "lr": 1.9e-6}}]}
    runs = board_runs_from_status(st)
    tw = next(r for r in runs if r["name"] == "train:olmoe-v5")
    assert tw["status"] == "running"
    assert tw["metrics"]["current_step"] == 100.0 and tw["metrics"]["total_steps"] == 220.0
    assert tw["metrics"]["loss"] == 0.54
    assert tw.get("steps", {}).get("total") == 220


def test_status_null_running_no_job() -> None:
    runs = board_runs_from_status({"running": None, "pendingCommands": []})
    assert runs == []


def test_determinism() -> None:
    v = {"mean_kl": 0.04, "top1_agreement": 0.95, "passed": True}
    idx = {"certify-lowram-olmoe-nvfp4-v3": v}
    assert board_runs_from_queue(FORECAST_QUEUE, idx) == board_runs_from_queue(FORECAST_QUEUE, idx)
    st = {"running": "x", "pendingCommands": [{"id": "y", "args": "--all"}]}
    assert board_runs_from_status(st) == board_runs_from_status(st)
    res = {"a": 1, "b": 2.0, "passed": True, "note": "x"}
    assert _result_run(res, "s") == _result_run(res, "s")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} trainwatch_benchmark_board tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

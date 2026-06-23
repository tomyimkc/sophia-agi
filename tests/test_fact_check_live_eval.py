#!/usr/bin/env python3
"""Offline measured eval tests for the out-of-wiki fact-check gate."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_eval import load_jsonl, run_fact_check_eval, wilson_interval  # noqa: E402
from agent.live_sources import FixtureFactBackend  # noqa: E402
from tools.run_fact_check_live_eval import main as cli_main  # noqa: E402

PACK = ROOT / "eval" / "fact_check" / "heldout_v1.jsonl"
FIXTURES = ROOT / "eval" / "fact_check" / "fixtures_v1.json"


def test_pack_has_n_ge_40_and_label_mix() -> None:
    rows = load_jsonl(PACK)
    assert len(rows) >= 40
    labels = {r["label"] for r in rows}
    assert {"true", "false", "unknowable"} <= labels


def test_offline_fact_check_eval_reports_core_metrics() -> None:
    rows = load_jsonl(PACK)
    b = FixtureFactBackend.from_file(FIXTURES)
    report = run_fact_check_eval(rows, retriever=b.retriever, entailment=b.entailment,
                                 doi_resolver=b.doi_resolver, url_resolver=b.url_resolver)
    assert report["candidateOnly"] is True
    assert report["n"] >= 40
    assert report["metrics"]["fabricationRate"] == 0.0
    assert report["metrics"]["overAbstentionRate"] <= 0.20
    assert report["metrics"]["calibrationNResolved"] > 0
    assert "normal" in report["derivedFloors"]["byRisk"]


def test_cli_writes_report() -> None:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "report.json"
        rc = cli_main(["--pack", str(PACK), "--fixtures", str(FIXTURES), "--out", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["schema"] == "sophia.fact_check.live_eval.v1"
        assert data["level3Evidence"] is False


def test_wilson_interval_shape() -> None:
    ci = wilson_interval(0, 25)
    assert ci["k"] == 0 and ci["n"] == 25
    assert 0.0 <= ci["low"] <= ci["high"] <= 1.0


def main() -> int:
    test_pack_has_n_ge_40_and_label_mix()
    test_offline_fact_check_eval_reports_core_metrics()
    test_cli_writes_report()
    test_wilson_interval_shape()
    print("test_fact_check_live_eval: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline coverage for the concept RLVR task end-to-end (no torch, no GPU).

Guards that the rlvr-runpod workflow actually works for task=concept: the CLI
mock/dry-run writes a passing report, the adapter-eval (mock) produces a report,
and that report ingests cleanly into the SSIL Layer-1 gate (the half that errored
before concept support was added to tools/eval_rlvr_adapter.py)."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import eval_rlvr_adapter, ingest_rlvr_eval, run_rlvr  # noqa: E402


def test_run_rlvr_mock_concept_passes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "rlvr-concept.json"
        code = run_rlvr.main(["--model", "mock", "--task", "concept", "--dry-run", "--out", str(out)])
        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["task"] == "concept"
        assert all(report["checks"].values()), report["checks"]
        # the concept task additionally gates on the spurious-reward ablation
        assert report["checks"]["spuriousAblationDiscriminates"] is True


def test_concept_adapter_eval_and_ingest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "concept.adapter-eval.json"
        code = eval_rlvr_adapter.main(["--mode", "mock", "--task", "concept", "--out", str(out)])
        assert code == 0
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["task"] == "concept"
        assert report["passed"] is True
        # the report carries the keys ingest_rlvr_eval.map_report needs
        assert "meanReward" in report["base"] and "meanReward" in report["adapterScore"]
        # adapter (grounded) must beat base (careless) and not raise over-abstention
        assert report["delta"]["meanReward"] > 0
        assert report["delta"]["overAbstainRate"] <= 0

        mapped = ingest_rlvr_eval.map_report(report)
        assert mapped["task"] == "concept"
        assert mapped["capabilityMetric"] == "meanReward"
        assert mapped["after"] > mapped["before"]
        assert mapped["contaminated"] is False


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            globals()[name]()
    print("concept RLVR end-to-end PASS")

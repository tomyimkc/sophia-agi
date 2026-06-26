# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the --capability-panel integration in tools/eval_rlvr_adapter.py.

Verifies the panel attaches as an additive key without disturbing the legacy
report (the SSIL/aggregate consumers read base/adapterScore/delta verbatim).
Deterministic, no torch, no GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_eval():
    spec = importlib.util.spec_from_file_location(
        "eval_rlvr_adapter", ROOT / "tools" / "eval_rlvr_adapter.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _load_ingest():
    spec = importlib.util.spec_from_file_location(
        "ingest_rlvr_eval", ROOT / "tools" / "ingest_rlvr_eval.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def _args(eval_mod, *, capability_panel: bool):
    import argparse

    ns = argparse.Namespace(
        mode="mock", task="provenance", model="zai-org/glm-4-9b-chat-hf",
        adapter=None, seed=0, eval_frac=0.3, limit=0, max_new_tokens=128,
        max_fp_regression=0.0, capability_panel=capability_panel,
    )
    return ns


# --------------------------------------------------------------------------- #
# Legacy keys are byte-identical whether or not the panel runs.
# --------------------------------------------------------------------------- #
def test_panel_is_additive_legacy_keys_unchanged():
    eval_mod = _load_eval()
    without = eval_mod.run_eval(_args(eval_mod, capability_panel=False))
    with_panel = eval_mod.run_eval(_args(eval_mod, capability_panel=True))
    # The panel key appears only when requested.
    assert "capabilityPanel" not in without
    assert "capabilityPanel" in with_panel
    # Every legacy key is identical between the two runs.
    for key in ("base", "adapterScore", "delta", "checks", "passed",
                "falsePositiveRegressions", "split", "benchmark", "claimStatus"):
        assert without[key] == with_panel[key], f"legacy key changed by panel: {key}"


def test_panel_report_shape():
    eval_mod = _load_eval()
    report = eval_mod.run_eval(_args(eval_mod, capability_panel=True))
    cp = report["capabilityPanel"]
    assert cp["schema"] == "sophia.capability_panel.v1"
    assert cp["candidateOnly"] is True
    assert cp["ok"] is True
    # The four advisor axes all appear in the embedded panel delta.
    for key in ("verdictAccuracy", "hallucinationRate", "integrityRecall", "calibrationScore"):
        assert key in cp["delta"]


def test_ingest_surfaces_panel_delta_when_present():
    ingest_mod = _load_ingest()
    eval_mod = _load_eval()
    report = eval_mod.run_eval(_args(eval_mod, capability_panel=True))
    mapped = ingest_mod.map_report(report)
    assert "capabilityPanelDelta" in mapped
    cpd = mapped["capabilityPanelDelta"]
    for key in ("verdictAccuracy", "hallucinationRate", "integrityRecall", "calibrationScore"):
        assert key in cpd


def test_ingest_fail_open_when_panel_absent():
    """An old report (no capabilityPanel) must still ingest exactly as before —
    the panel is additive evidence, not a gate input."""
    ingest_mod = _load_ingest()
    eval_mod = _load_eval()
    report = eval_mod.run_eval(_args(eval_mod, capability_panel=False))
    mapped = ingest_mod.map_report(report)
    assert "capabilityPanelDelta" not in mapped
    # The legacy headline numbers are present and well-formed.
    assert mapped["before"] == report["base"]["meanReward"]
    assert mapped["after"] == report["adapterScore"]["meanReward"]


def test_ingest_real_committed_report_still_maps():
    """The committed runpod-rlvr adapter-eval report (no panel) must still map
    cleanly through the (now panel-aware) ingest."""
    ingest_mod = _load_ingest()
    import json

    committed = ROOT / "agi-proof" / "benchmark-results" / "runpod-rlvr" / "mr9sr03clgpk5g.rlvr.adapter-eval.json"
    if not committed.exists():
        pytest.skip("committed adapter-eval report not present")
    mapped = ingest_mod.map_report(json.loads(committed.read_text(encoding="utf-8")))
    assert "capabilityPanelDelta" not in mapped
    assert mapped["before"] < mapped["after"]  # the documented capability gain


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

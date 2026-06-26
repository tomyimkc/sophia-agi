# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the capability-delta panel (tools/eval_capability_panel.py).

All deterministic, no torch, no GPU, no network. Exercises the mock-mode
end-to-end path and the invariants the report must satisfy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_panel():
    spec = importlib.util.spec_from_file_location(
        "eval_capability_panel", ROOT / "tools" / "eval_capability_panel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


panel = _load_panel()


def _run_mock(limit: int = 0):
    """Run the panel in mock mode without writing a public artifact."""
    base_attr, adapter_attr, cal_base, cal_adapter = panel._mock_generators(limit)
    cases = panel._load_attribution_cases(limit)
    cal_pack = panel._load_calibration_pack(limit)
    attr = panel._attribution_axis(cases, base_attr, adapter_attr)
    cal = panel._calibration_axis(cal_pack, cal_base, cal_adapter)
    return panel._deltas(attr, cal), attr, cal


# --------------------------------------------------------------------------- #
# Report shape + honesty fields
# --------------------------------------------------------------------------- #
def test_report_has_required_honesty_fields(tmp_path):
    report = panel.run(mode="mock", limit=20, out=tmp_path / "panel.json")
    # The no-overclaim honesty fields every Sophia benchmark carries.
    for key in ("schema", "benchmark", "claimStatus", "candidateOnly",
                "level3Evidence", "validated", "claimBoundary", "ok", "passed"):
        assert key in report, f"missing honesty field: {key}"
    assert report["schema"] == "sophia.capability_panel.v1"
    assert report["candidateOnly"] is True
    assert report["level3Evidence"] is False
    assert report["validated"] is False
    assert "Open" in report["claimStatus"]


def test_axes_present(tmp_path):
    report = panel.run(mode="mock", limit=20, out=tmp_path / "panel.json")
    assert set(report["axes"]) >= {"attribution", "calibration"}
    for axis in ("attribution",):
        for arm in ("base", "adapter"):
            assert arm in report["axes"][axis]
    # The four advisor-cited metrics all appear in the report somewhere.
    flat = report["delta"]
    for key in ("verdictAccuracy", "hallucinationRate", "integrityRecall", "calibrationScore"):
        assert key in flat


# --------------------------------------------------------------------------- #
# Invariants: an improved arm must look improved on every axis.
# --------------------------------------------------------------------------- #
def test_mock_adapter_improves_on_every_axis():
    deltas, attr, cal = _run_mock(limit=0)
    # The mock adapter is a deliberately-better answerer, so the deltas must
    # point the right way on all four axes.
    assert deltas["verdictAccuracy"] > 0          # capability went up
    assert deltas["hallucinationRate"] < 0        # harm went down
    assert deltas["integrityRecall"] > 0          # integrity went up
    assert deltas["calibrationScore"] > 0         # calibration went up
    assert deltas["fabricationRate"] < 0          # fabrication went down


def test_base_arm_hallucinates_on_false_cases():
    _, attr, _ = _run_mock(limit=0)
    base = attr["base"]
    # The mock base asserts the forbidden attribution on false cases, so its
    # integrityRecall must be low (it certified what it should not have).
    assert base["integrity"]["integrityRecall"] is not None
    assert base["integrity"]["integrityRecall"] < 0.2
    assert base["hallucinationRate"] > 0.5


def test_adapter_arm_abstains_on_false_cases():
    _, attr, _ = _run_mock(limit=0)
    adapter = attr["adapter"]
    # The mock adapter abstains / corrects, so it must NOT certify the forbidden.
    assert adapter["integrity"]["certifiedForbidden"] == 0
    assert adapter["hallucinationRate"] == 0.0
    assert adapter["integrity"]["integrityRecall"] == 1.0


def test_calibration_fabrication_drops_to_zero_on_adapter():
    _, _, cal = _run_mock(limit=0)
    # Base fabricates (asserts the claimed author); adapter abstains honestly.
    assert cal["base"]["fabricationRate"] > 0
    assert cal["adapter"]["fabricationRate"] == 0.0


# --------------------------------------------------------------------------- #
# Structural / numeric invariants
# --------------------------------------------------------------------------- #
def test_n_counts_are_consistent():
    deltas, attr, cal = _run_mock(limit=0)
    assert attr["n"] > 0
    assert cal["n"] > 0
    # Both arms scored the same number of cases.
    assert attr["base"]["integrity"]["falseCases"] == attr["adapter"]["integrity"]["falseCases"]


def test_deltas_are_base_to_adapter_signed():
    deltas, attr, _ = _run_mock(limit=0)
    # delta == adapter - base, by construction (sanity-check one).
    expected = round(attr["adapter"]["verdictAccuracy"] - attr["base"]["verdictAccuracy"], 4)
    assert deltas["verdictAccuracy"] == pytest.approx(expected, abs=1e-4)


def test_calib_pack_maps_seib_labels_to_abstain():
    pack = panel._load_calibration_pack(limit=10)
    assert pack["cases"]
    for case in pack["cases"]:
        # SEIB's false_attribution + qualify_or_abstain both map to abstain-type.
        assert case["epistemicLabel"] == "abstain"
    # At least one false_attribution case carries fabricationMarkers (the
    # claimed-author-derived fabrication signal).
    assert any(case.get("fabricationMarkers") for case in pack["cases"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

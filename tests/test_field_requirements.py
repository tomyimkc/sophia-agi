#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The field-requirements capability proof must stay honest: every artifact the
manifest cites must exist and compile. This test makes that a CI gate — delete a
cited module or report and the proof FAILS here, rather than silently lying."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "agi-proof" / "field-requirements" / "manifest.json"


def _verifier():
    spec = importlib.util.spec_from_file_location(
        "vfr", ROOT / "tools" / "verify_field_requirements.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_all_cited_artifacts_exist_and_compile() -> None:
    report = _verifier().verify(_manifest())
    bad = [c["id"] for c in report["capabilities"] if not c["ok"]]
    assert report["allOk"], f"capabilities with missing/uncompilable artifacts: {bad}"
    assert report["capabilitiesTotal"] >= 8


def test_every_capability_has_module_test_and_evidence() -> None:
    for cap in _manifest()["capabilities"]:
        assert cap.get("modules"), f"{cap['id']} cites no module"
        assert cap.get("tests"), f"{cap['id']} cites no test"
        assert cap.get("evidence"), f"{cap['id']} cites no evidence"


def test_statuses_are_from_the_legend() -> None:
    manifest = _manifest()
    legend = set(manifest["statusLegend"])
    for cap in manifest["capabilities"]:
        assert cap["status"] in legend, f"{cap['id']} has off-legend status {cap['status']!r}"


def test_demonstrated_capabilities_carry_a_measured_line() -> None:
    # 'demonstrated' is the strongest tier; it must point at something measured.
    for cap in _manifest()["capabilities"]:
        if cap["status"] == "demonstrated":
            assert cap.get("measured"), f"{cap['id']} is demonstrated but has no 'measured' line"


def test_agent_data_evaluation_is_present_and_demonstrated() -> None:
    # The headline market-fit capability (the new "Agent Data Evaluation" role).
    caps = {c["id"]: c for c in _manifest()["capabilities"]}
    cap = caps.get("agent-data-evaluation")
    assert cap is not None
    assert cap["status"] == "demonstrated"
    assert "agent/trajectory_eval.py" in cap["modules"]


def test_manifest_cites_the_market_source() -> None:
    src = _manifest().get("marketSource", {})
    assert src.get("citations"), "market source must carry citations"
    assert any("deepseek" in u.lower() for u in src["citations"])


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the G4W weight-space / LoRA-delta audit gate. Offline, no torch.

Covers: a promote path via demo_bundle(); the reject reason (out-of-scope module);
each quarantine reason (magnitude spike, localized outlier spike, and each
fail-closed missing-input branch); and the standing invariants (canClaimAGI False,
candidateOnly True, verdict in the allowed set).
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_weight_audit import (  # noqa: E402
    GATE_ID,
    SCHEMA,
    demo_bundle,
    evaluate,
)

_ALLOWED = {"promote", "quarantine", "reject"}


def test_demo_bundle_promotes() -> None:
    d = evaluate(demo_bundle())
    assert d["verdict"] == "promote", d["reasons"]
    assert d["gate"] == GATE_ID
    assert d["schema"] == SCHEMA
    assert d["metrics"]["outOfScopeModules"] == []


def test_reject_out_of_scope_module() -> None:
    """A changed module outside intendedModules -> reject (possible backdoor)."""
    b = copy.deepcopy(demo_bundle())
    b["deltaStats"]["perModuleNorm"]["model.embed_tokens"] = 0.5  # not in intendedModules
    d = evaluate(b)
    assert d["verdict"] == "reject", d["reasons"]
    assert any("possible backdoor" in r for r in d["reasons"])
    assert "model.embed_tokens" in d["metrics"]["outOfScopeModules"]


def test_quarantine_magnitude_spike() -> None:
    """maxSingularValue above the ceiling -> quarantine (anomalous weight spike)."""
    b = copy.deepcopy(demo_bundle())
    b["deltaStats"]["maxSingularValue"] = 5.0
    b["maxAllowedSingular"] = 1.0
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("anomalous weight spike" in r for r in d["reasons"])


def test_quarantine_localized_outlier_spike() -> None:
    """One module norm far above the median of the rest -> quarantine (trojan)."""
    b = copy.deepcopy(demo_bundle())
    b["deltaStats"]["perModuleNorm"]["model.layers.0.self_attn.q_proj"] = 100.0
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("localized spike: possible trojan" in r for r in d["reasons"])
    assert d["metrics"]["spikeModule"] == "model.layers.0.self_attn.q_proj"


def test_fail_closed_missing_delta_stats() -> None:
    b = copy.deepcopy(demo_bundle())
    b.pop("deltaStats")
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("deltaStats" in r for r in d["reasons"])


def test_fail_closed_missing_per_module_norm() -> None:
    b = copy.deepcopy(demo_bundle())
    b["deltaStats"].pop("perModuleNorm")
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("perModuleNorm" in r for r in d["reasons"])


def test_fail_closed_missing_intended_modules() -> None:
    b = copy.deepcopy(demo_bundle())
    b.pop("intendedModules")
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("intendedModules" in r for r in d["reasons"])


def test_fail_closed_missing_max_allowed_singular() -> None:
    b = copy.deepcopy(demo_bundle())
    b.pop("maxAllowedSingular")
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("maxAllowedSingular" in r for r in d["reasons"])


def test_fail_closed_missing_max_singular_value() -> None:
    b = copy.deepcopy(demo_bundle())
    b["deltaStats"].pop("maxSingularValue")
    d = evaluate(b)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("maxSingularValue" in r for r in d["reasons"])


def test_fail_closed_bundle_none() -> None:
    d = evaluate(None)
    assert d["verdict"] == "quarantine", d["reasons"]
    assert any("bundle is None" in r for r in d["reasons"])


def test_standing_invariants() -> None:
    """Every decision: canClaimAGI False, candidateOnly True, level3Evidence False,
    verdict in the allowed set, and an honest non-empty boundary."""
    bundles = [
        demo_bundle(),
        None,
        {"deltaStats": {"perModuleNorm": {"a": 0.1, "b": 9.9},
                        "maxSingularValue": 0.2, "rank": 4, "sparsity": 0.0},
         "intendedModules": ["a", "b"], "maxAllowedSingular": 1.0},
    ]
    for b in bundles:
        d = evaluate(b, candidate_id="g4w-test")
        assert d["canClaimAGI"] is False
        assert d["candidateOnly"] is True
        assert d["level3Evidence"] is False
        assert d["verdict"] in _ALLOWED
        assert isinstance(d["boundary"], str) and d["boundary"]
        assert d["candidateId"] == "g4w-test"


def main() -> int:
    test_demo_bundle_promotes()
    test_reject_out_of_scope_module()
    test_quarantine_magnitude_spike()
    test_quarantine_localized_outlier_spike()
    test_fail_closed_missing_delta_stats()
    test_fail_closed_missing_per_module_norm()
    test_fail_closed_missing_intended_modules()
    test_fail_closed_missing_max_allowed_singular()
    test_fail_closed_missing_max_singular_value()
    test_fail_closed_bundle_none()
    test_standing_invariants()
    print("test_ssil_weight_audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

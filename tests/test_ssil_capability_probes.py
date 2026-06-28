#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the dangerous-capability probe protocol (G8 input layer). Offline."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import ssil_capability_ceiling as g8  # noqa: E402
from agent import ssil_capability_probes as probes  # noqa: E402


def test_shipped_registry_is_unprovisioned_failclosed() -> None:
    probes.reset_registry()
    assert probes.is_g8_provisioned() is False
    out = probes.probe_scores({})
    assert out["allMeasured"] is False
    assert set(out["scores"]) == set(g8.REQUIRED_PROBES)
    assert all(v is None for v in out["scores"].values())
    assert sorted(out["unvalidated"]) == sorted(g8.REQUIRED_PROBES)


def test_g8_quarantines_on_shipped_placeholders() -> None:
    probes.reset_registry()
    decision = g8.evaluate(probes.g8_bundle({}))
    assert decision["verdict"] == "quarantine"  # unmeasured danger cannot be cleared
    assert decision["canClaimAGI"] is False and decision["candidateOnly"] is True


def test_registered_validated_probes_make_g8_enforceable() -> None:
    probes.reset_registry()
    # Register a validated probe per capability that reads the pod-supplied score.
    for name in g8.REQUIRED_PROBES:
        probes.register_probe(name, (lambda n: lambda rep: rep.get("probeScores", {}).get(n))(name), validated=True)
    assert probes.is_g8_provisioned() is True
    clean = {"probeScores": probes.synthetic_scores("clean")}
    out = probes.probe_scores(clean)
    assert out["allMeasured"] is True and out["provisioned"] is True
    assert g8.evaluate(probes.g8_bundle(clean))["verdict"] == "promote"
    probes.reset_registry()


def test_halt_scores_reject_through_g8() -> None:
    probes.reset_registry()
    for name in g8.REQUIRED_PROBES:
        probes.register_probe(name, (lambda n: lambda rep: rep.get("probeScores", {}).get(n))(name), validated=True)
    halt = {"probeScores": probes.synthetic_scores("halt")}
    decision = g8.evaluate(probes.g8_bundle(halt))
    assert decision["verdict"] == "reject"
    assert decision["metrics"].get("halt") is True
    probes.reset_registry()


def test_unvalidated_probe_does_not_contribute() -> None:
    probes.reset_registry()
    # Implemented but NOT validated -> must stay unmeasured (cannot certify danger away).
    probes.register_probe("cyber_offense", lambda rep: 0.05, validated=False)
    out = probes.probe_scores({})
    assert out["scores"]["cyber_offense"] is None
    assert probes.is_g8_provisioned() is False
    probes.reset_registry()


def test_crashing_probe_is_failclosed_unmeasured() -> None:
    probes.reset_registry()

    def boom(_rep):
        raise RuntimeError("probe blew up")

    probes.register_probe("deception", boom, validated=True)
    out = probes.probe_scores({})
    assert out["scores"]["deception"] is None  # crash -> unmeasured, never a passing score
    probes.reset_registry()


def test_synthetic_scores_levels() -> None:
    assert probes.synthetic_scores("clean")["cyber_offense"] < g8.WARN_THRESHOLD_DEFAULT
    assert g8.WARN_THRESHOLD_DEFAULT <= probes.synthetic_scores("warn")["cyber_offense"] < g8.HALT_THRESHOLD_DEFAULT
    assert probes.synthetic_scores("halt")["cyber_offense"] >= g8.HALT_THRESHOLD_DEFAULT
    try:
        probes.synthetic_scores("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown level must raise")


def main() -> int:
    test_shipped_registry_is_unprovisioned_failclosed()
    test_g8_quarantines_on_shipped_placeholders()
    test_registered_validated_probes_make_g8_enforceable()
    test_halt_scores_reject_through_g8()
    test_unvalidated_probe_does_not_contribute()
    test_crashing_probe_is_failclosed_unmeasured()
    test_synthetic_scores_levels()
    print("test_ssil_capability_probes: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

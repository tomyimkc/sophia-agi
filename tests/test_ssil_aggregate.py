#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.ssil_aggregate import AdapterAggregate, SeedRun, harden_verdict  # noqa: E402
from agent.ssil_registry import Registry  # noqa: E402

CONFIG = {"adapter": "sophia-rlvr-v1", "kind": "lora_adapter"}


def _three_good() -> list[SeedRun]:
    return [
        SeedRun(0, before=0.531, after=0.7149, protected_before=0.7917, protected_after=0.7917),
        SeedRun(1, before=0.540, after=0.690, protected_before=0.79, protected_after=0.80),
        SeedRun(2, before=0.520, after=0.705, protected_before=0.79, protected_after=0.79),
    ]


def test_harden_promotes_clean_gain() -> None:
    v, reasons = harden_verdict(SeedRun(0, 0.53, 0.71, 0.79, 0.79))
    assert v == "promote", reasons


def test_harden_rejects_small_gain() -> None:
    v, reasons = harden_verdict(SeedRun(0, 0.70, 0.71, 0.79, 0.79))  # +0.01 < 0.03
    assert v == "reject" and any("below floor" in r for r in reasons)


def test_harden_rejects_protected_regression() -> None:
    v, reasons = harden_verdict(SeedRun(0, 0.53, 0.71, 0.79, 0.70))  # integrity dropped
    assert v == "reject" and any("integrity regressed" in r for r in reasons)


def test_harden_rejects_contamination() -> None:
    v, reasons = harden_verdict(SeedRun(0, 0.53, 0.71, 0.79, 0.79, contaminated=True))
    assert v == "reject" and any("contaminated" in r for r in reasons)


def test_harden_against_canonical_baseline() -> None:
    # after=0.71 beats base (0.53) but only +0.01 over canonical 0.70 -> not an improvement.
    v, reasons = harden_verdict(SeedRun(3, 0.53, 0.71, 0.79, 0.79), baseline_after=0.70)
    assert v == "reject" and any("over canonical" in r for r in reasons)


def test_aggregate_claim_ready() -> None:
    agg = AdapterAggregate("sophia-rlvr-v1", CONFIG, _three_good(), canonical_n=3)
    s = agg.summary()
    assert s["n"] == 3 and s["promotes"] == 3
    assert s["capability"]["meanDelta"] > 0
    assert s["capabilityClaimReady"] is True
    assert s["canClaimAGI"] is False and s["candidateOnly"] is True


def test_aggregate_not_ready_with_two_seeds() -> None:
    agg = AdapterAggregate("sophia-rlvr-v1", CONFIG, _three_good()[:2], canonical_n=3)
    assert agg.summary()["capabilityClaimReady"] is False  # n < canonical_n


def test_registry_canonical_after_n_and_baseline_to_beat() -> None:
    reg = Registry(path=None, canonical_n=3)
    agg = AdapterAggregate("sophia-rlvr-v1", CONFIG, _three_good(), canonical_n=3)
    rec = agg.record_to_registry(reg)
    assert rec["replications"] == 3
    assert rec["canonical"] is True
    assert rec["nextAdapterMustBeat"] is not None  # the canonical mean after


def main() -> int:
    test_harden_promotes_clean_gain()
    test_harden_rejects_small_gain()
    test_harden_rejects_protected_regression()
    test_harden_rejects_contamination()
    test_harden_against_canonical_baseline()
    test_aggregate_claim_ready()
    test_aggregate_not_ready_with_two_seeds()
    test_registry_canonical_after_n_and_baseline_to_beat()
    print("test_ssil_aggregate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

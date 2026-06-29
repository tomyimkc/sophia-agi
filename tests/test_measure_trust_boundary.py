#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""On-vs-off contamination measurement for the verifier-gated trust boundary.

Deterministic and offline — the gate runs the repo's machine verifiers, no model, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.measure_trust_boundary import measure, offline_invariants  # noqa: E402


def test_offline_invariants_pass() -> None:
    ok, detail = offline_invariants()
    assert ok, detail["checks"]


def test_boundary_blocks_all_detectable_contamination() -> None:
    m = measure()
    assert m["contaminationRateOff"] == 1.0   # no gate -> all detectable poison reaches siblings
    assert m["contaminationRateOn"] == 0.0    # boundary -> none does
    assert m["contaminationBlockedRate"] == 1.0


def test_residual_is_reported_not_hidden() -> None:
    # The gate is a filter, not a truth oracle: the measurement must expose any poison that
    # carried no detectable violation rather than imply perfect coverage.
    m = measure()
    assert "admittedPoisonResidual" in m
    assert m["verifierFlagged"] >= 1


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok {t.__name__}")
    print(f"PASS {len(tests)} measure_trust_boundary tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

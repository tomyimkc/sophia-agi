#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline-safe tests for the Google Fact Check coverage probe.

The live coverage numbers are non-deterministic (real API), so CI only asserts the
fail-closed contract: with no API key the backend returns no evidence and the probe
reports zero coverage — never a crash, never fabricated coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.live_sources import GoogleFactCheckBackend
from tools.run_google_factcheck_coverage import _coverage, _GENERAL, _PROVENANCE


def test_no_key_fails_closed_zero_coverage():
    backend = GoogleFactCheckBackend(api_key="")
    assert backend.api_key == ""
    cov = _coverage(backend, _GENERAL)
    assert cov["covered"] == 0
    assert cov["coverageRate"] == 0.0
    assert cov["n"] == len(_GENERAL)


def test_claim_sets_are_disjoint_and_nonempty():
    assert _GENERAL and _PROVENANCE
    assert not (set(_GENERAL) & set(_PROVENANCE))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

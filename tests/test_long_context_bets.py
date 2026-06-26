#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate the long-context measurement-target registry.

This is the 7-bet ``honest_status`` / ``blocked_on`` / ``implementation_files`` registry
that previously shared the ``agi-proof/architecture-bets.json`` filename with the
module-wiring governance registry. It was relocated to
``agi-proof/long-context-bets.json`` so the two incompatible schemas can coexist; see
docs/11-Platform/Architecture-Bets-Schema.md.

These are exactly the invariants that
``tests/test_long_context_runner.py::test_architecture_bets_root_map_has_required_fields``
asserted, retargeted at the new file. When the
``claude/sophia-agi-architecture-review-ucvzyl`` branch lands, that test should drop its
own copy of these assertions (one-line path retarget) in favour of this test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY = ROOT / "agi-proof" / "long-context-bets.json"

REQUIRED_BET_IDS = {
    "verifier-gated-long-context",
    "hybrid-memory",
    "selective-tool-router",
    "council-small-models",
    "verifier-as-reward",
    "long-context-compression-recall",
    "ablation-harness",
}
VALID_HONEST_STATUS = {"scaffold", "partial", "live"}


def _load() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_long_context_bets_root_map_has_required_fields() -> None:
    bets = _load()
    assert bets["candidateOnly"] is True
    assert bets["canClaimAGI"] is False
    by_id = {bet["id"]: bet for bet in bets["bets"]}
    assert set(by_id) == REQUIRED_BET_IDS
    for bet in by_id.values():
        assert bet["implementation_files"]
        assert bet["ablation_flag"]
        assert bet["honest_status"] in VALID_HONEST_STATUS
        assert "blocked_on" in bet


def main() -> int:
    test_long_context_bets_root_map_has_required_fields()
    print("test_long_context_bets: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

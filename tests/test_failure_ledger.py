# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""The failure ledger is the most honest artifact in the repo — keep it from rotting.

Validates agi-proof/failure-ledger.md structurally (every table entry has an id, Status,
and Claim impact) and asserts the OPEN/CLOSED summary is non-degenerate. Dependency-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_failure_ledger import LEDGER, classify, validate  # noqa: E402


def test_ledger_exists_and_is_valid() -> None:
    assert LEDGER.exists(), "agi-proof/failure-ledger.md must exist"
    result = validate(LEDGER)
    assert result["ok"], f"failure ledger structurally invalid: {result['missing']}"
    assert result["tableRows"] >= 10, result  # the ledger is non-degenerate


def test_status_classifier() -> None:
    assert classify("Open") == "open"
    assert classify("PARTIAL (real progress)") == "open"
    assert classify("BLOCKED (ssh egress)") == "open"
    assert classify("Closed") == "resolved"
    assert classify("Cleared (rung)") == "resolved"
    assert classify("Falsified") == "resolved"
    assert classify("Superseded") == "resolved"
    assert classify("") == "other"


def test_open_items_are_real_failure_ids() -> None:
    result = validate(LEDGER)
    # every OPEN item should be a kebab-case id (optionally with a date suffix)
    for item in result["openItems"]:
        assert "-" in item and " " not in item, f"malformed OPEN id: {item!r}"


def test_summary_has_open_count() -> None:
    result = validate(LEDGER)
    # there must be at least one OPEN item (the AGI claim is not proven) — if this ever
    # flips to 0, the public wording must be upgraded, not this assertion silently relaxed.
    assert result["byStatus"]["open"] >= 1, "expected >=1 OPEN failure-ledger item"


def main() -> int:
    import inspect
    for nm, fn in sorted(globals().items()):
        if nm.startswith("test_") and inspect.isfunction(fn):
            fn()
    print("test_failure_ledger: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the belief-dynamics wiring into the offline CLS consolidation selector.

Verifies the narrow, honest wiring of Option B:
  - the tamper-evident ForgettingAudit ledger records every consolidation selection from
    a real (temp-wiki) run, and the chain verifies;
  - the selected set is UNCHANGED by the dynamics (the anti-forgetting gate stays the
    floor downstream);
  - the projection is loudly honest: the unrecorded signals are the documented
    placeholders, and level3Evidence stays false.

Offline, deterministic, dependency-free (synthetic pages in a temp dir).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from okf import frontmatter  # noqa: E402
from okf.belief_state_projection import (  # noqa: E402
    GENESIS_EPOCH, UNRECORDED_REINFORCEMENT, UNRECORDED_SURPRISE,
)
from tools.run_cls_consolidation import build_selection  # noqa: E402

_RICH_BODY = ("# analects (論語)\n\nThe Analects is a compiled record of conversations "
              "attributed to Confucius and his disciples.\n")


def _write(dirpath, pid, body):
    meta = {"id": pid, "pageType": "concept", "authorConfidence": "consensus"}
    (Path(dirpath) / f"{pid}.md").write_text(frontmatter.serialize(meta, body), encoding="utf-8")


def test_audit_ledger_records_every_selection_and_verifies() -> None:
    """A real run lands every selected fact in the tamper-evident chain; chain verifies."""
    with tempfile.TemporaryDirectory() as d:
        _write(d, "analects", _RICH_BODY)
        sel = build_selection(d, min_stable_snapshots=1)
        assert sel["selected"] == ["analects"]
        assert sel["auditChainValid"] is True
        # one consolidate event per selected fact
        events = [e for e in sel["auditChain"] if e["event"] == "consolidate"]
        assert len(events) == 1
        assert events[0]["node_id"] == "analects"
        assert events[0]["reason"] == "cls_selection"
        # the genesis record links to the all-zeros prev_hash
        assert sel["auditChain"][0]["prev_hash"] == "0" * 64


def test_dynamics_does_not_change_the_selected_set() -> None:
    """The forgetting layer records the selection; it does not alter WHAT is selected.
    The anti-forgetting gate (evaluate_update) remains the floor downstream."""
    with tempfile.TemporaryDirectory() as d:
        _write(d, "analects", _RICH_BODY)
        sel = build_selection(d, min_stable_snapshots=1)
        # the decay plan never deletes (forgetting is demotion), and with unrecorded
        # timestamps (all GENESIS_EPOCH) time-decay is a no-op: nothing is suppressed.
        assert sel["decayPlan"]["deletions"] == 0
        assert sel["decayPlan"]["suppress"] == []
        assert sel["selected"] == ["analects"]   # unchanged by dynamics


def test_projection_is_loudly_honest_about_unrecorded_signals() -> None:
    """The documented placeholders are exactly what the projection uses; level3 stays false."""
    with tempfile.TemporaryDirectory() as d:
        _write(d, "analects", _RICH_BODY)
        sel = build_selection(d, min_stable_snapshots=1)
        ph = sel["unrecordedPlaceholders"]
        assert ph == {"writtenAtEpoch": GENESIS_EPOCH,
                      "surprise": UNRECORDED_SURPRISE,
                      "reinforcementCount": UNRECORDED_REINFORCEMENT}
        # the honesty note must name the placeholders explicitly
        assert "UNRECORDED PLACEHOLDERS" in sel["projectionHonestyNote"]
        assert "unmeasured, not" in sel["projectionHonestyNote"]   # "unsurprising"/"never used"
        assert sel["decayPlan"]["level3Evidence"] is False


def main() -> int:
    test_audit_ledger_records_every_selection_and_verifies()
    test_dynamics_does_not_change_the_selected_set()
    test_projection_is_loudly_honest_about_unrecorded_signals()
    print("test_cls_consolidation_audit_wiring: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

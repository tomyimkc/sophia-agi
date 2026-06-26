#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for tools.eval_okf_surprise — the surprise-signal evidence driver.

Verifies the three evidence panels over the real wiki corpus:
  - SEPARATION: planted duplicate < planted novelty (the signal is discriminating).
  - CORPUS RUN: the measured signal flags a real, non-empty, auditable set the placeholder
    (surprise=0) could not — and the audit chain verifies.
  - GATE IS THE FLOOR: a surprise-selected candidate with no verified eval is NOT promoted.
And the level-3 determination: earned ONLY when all three pass; canClaimAGI stays false.

Offline, deterministic. Run: python tests/test_eval_okf_surprise.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_okf_surprise import run  # noqa: E402

WIKI = ROOT / "wiki"


def test_all_panels_pass() -> None:
    report = run(WIKI)
    panels = {p["panel"]: p for p in report["panels"]}
    assert panels["separation"]["pass"] is True
    assert panels["corpus-run"]["pass"] is True
    assert panels["gate-is-the-floor"]["pass"] is True
    assert report["pass"] is True


def test_separation_duplicate_below_novel() -> None:
    sep = {p["panel"]: p for p in run(WIKI)["panels"]}["separation"]
    assert sep["duplicate"]["rawNll"] < sep["novel"]["rawNll"]
    assert sep["duplicate"]["surprise"] < sep["novel"]["surprise"]


def test_measured_beats_placeholder_and_audit_verifies() -> None:
    cr = {p["panel"]: p for p in run(WIKI)["panels"]}["corpus-run"]
    assert cr["placeholderNovelCount"] == 0          # placeholder flags nothing
    assert cr["measuredNovelCount"] > 0              # measured flags a real set
    assert cr["auditChainValid"] is True
    assert cr["auditRecordCount"] == cr["measuredNovelCount"]


def test_gate_refuses_unverified_surprise_selection() -> None:
    gate = {p["panel"]: p for p in run(WIKI)["panels"]}["gate-is-the-floor"]
    assert gate["promoted"] is False
    assert gate["verdict"] != "promote"


def test_level3_scoped_and_agi_false() -> None:
    report = run(WIKI)
    # level3 is earned (all panels pass) but tightly scoped, and AGI is never claimed.
    assert report["level3Evidence"] is True
    assert report["canClaimAGI"] is False
    assert "Scoped to the SURPRISE signal" in report["level3Scope"]
    # the leave-one-out honesty caveat must be present and explicit.
    assert any("LEAVE-ONE-OUT" in c for c in report["honestyCaveats"])


def main() -> int:
    test_all_panels_pass()
    test_separation_duplicate_below_novel()
    test_measured_beats_placeholder_and_audit_verifies()
    test_gate_refuses_unverified_surprise_selection()
    test_level3_scoped_and_agi_false()
    print("test_eval_okf_surprise: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

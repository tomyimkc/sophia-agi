#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Validate the architecture-bets registry (W0).

The registry (`agi-proof/architecture-bets.json`) is the single source of truth for
which AGI-shaped module is wired onto the live `run_case` path vs still a scaffold.
These tests keep it honest: it must never claim AGI, every named module must resolve
to a real file, a bet may only be ``wired`` if it names a concrete ``live_caller``,
and ablation flags must be unique so the matrix can A/B each bet independently.

NOTE: the separate long-context measurement-target registry
(``agi-proof/long-context-bets.json``, fields honest_status/blocked_on/implementation_files)
is validated by ``tests/test_long_context_bets.py``. The two registries were deliberately
split so neither file has to pretend to be two schemas; see
docs/11-Platform/Architecture-Bets-Schema.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REGISTRY = ROOT / "agi-proof" / "architecture-bets.json"
VALID_STATUSES = {"scaffold", "wired", "measured", "retired"}


def _load() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_registry_parses_and_never_claims_agi() -> None:
    data = _load()
    assert data["canClaimAGI"] is False
    assert isinstance(data.get("bets"), list) and data["bets"], "registry must list bets"


def test_every_module_path_exists() -> None:
    data = _load()
    for bet in data["bets"]:
        module = bet["module"]
        assert (ROOT / module).exists(), f"bet {bet['id']}: module {module} does not exist"


def test_status_values_are_known() -> None:
    data = _load()
    for bet in data["bets"]:
        assert bet["status"] in VALID_STATUSES, f"bet {bet['id']}: bad status {bet['status']}"


def test_wired_bets_have_live_caller_whose_file_exists() -> None:
    data = _load()
    for bet in data["bets"]:
        if bet["status"] == "wired":
            caller = bet["live_caller"]
            assert caller, f"bet {bet['id']}: status=wired requires a non-null live_caller"
            # live_caller is "file:fn" — assert the file part exists on disk.
            caller_file = caller.split(":", 1)[0]
            assert (ROOT / caller_file).exists(), (
                f"bet {bet['id']}: live_caller file {caller_file} does not exist"
            )


def test_scaffold_bets_have_null_live_caller() -> None:
    data = _load()
    for bet in data["bets"]:
        if bet["status"] == "scaffold":
            assert bet["live_caller"] is None, (
                f"bet {bet['id']}: status=scaffold must have a null live_caller"
            )


def test_ablation_flag_unique_per_bet() -> None:
    data = _load()
    flags = [bet["ablation_flag"] for bet in data["bets"]]
    assert len(flags) == len(set(flags)), f"ablation_flag values must be unique: {flags}"


def test_required_fields_present() -> None:
    data = _load()
    required = {"id", "module", "status", "live_caller", "ablation_flag", "closing_experiment", "ledger_id"}
    ids = [bet["id"] for bet in data["bets"]]
    assert len(ids) == len(set(ids)), f"bet ids must be unique: {ids}"
    for bet in data["bets"]:
        missing = required - set(bet)
        assert not missing, f"bet {bet.get('id')}: missing fields {missing}"


def main() -> int:
    test_registry_parses_and_never_claims_agi()
    test_every_module_path_exists()
    test_status_values_are_known()
    test_wired_bets_have_live_caller_whose_file_exists()
    test_scaffold_bets_have_null_live_caller()
    test_ablation_flag_unique_per_bet()
    test_required_fields_present()
    print("test_architecture_bets: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

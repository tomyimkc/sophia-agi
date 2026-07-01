#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Drift + faithfulness tests for the Obsidian vault workflow diagram generator.

The committed note (docs/09-Agent/Vault-Workflow.md) must equal what the generator
produces, and every verdict the gate can emit must be routed in the diagram exactly
as route_after_verify() routes it — so the picture can never lie about the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_contract.langgraph_nodes import route_after_verify  # noqa: E402
from tools import vault_workflow_diagram as vwd  # noqa: E402


def test_committed_note_matches_generator():
    """The checked-in note is exactly what render() produces (no hand-edits)."""
    assert vwd.DEFAULT_OUT.exists(), "run: python tools/vault_workflow_diagram.py"
    assert vwd.DEFAULT_OUT.read_text(encoding="utf-8") == vwd.render()


def test_check_mode_passes():
    """--check exits 0 when the committed note is in sync."""
    assert vwd.main(["--check"]) == 0


def test_routes_are_code_derived():
    """Every verdict is routed in the note exactly as route_after_verify routes it."""
    routes = vwd._routes()
    for verdict, _ in vwd.VERDICTS:
        assert routes[verdict] == route_after_verify({"verdict": verdict})
        assert routes[verdict] in vwd.ROUTE_LABEL


def test_note_contains_mermaid_and_key_nodes():
    """The rendered note has Mermaid blocks and names the real entry points."""
    note = vwd.render()
    assert note.count("```mermaid") == 2
    for token in ("VaultGate.gate_note()", "record_claim", "verify_claim",
                  "route_after_verify", "publish_if_accepted", "CopywritingPipeline"):
        assert token in note


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")

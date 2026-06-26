#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tool-use/MCP curriculum metadata and JSONL shape tests (offline)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import build_local_sophia_dataset as build  # noqa: E402
from tools import validate_tool_use_curriculum as validate  # noqa: E402


def test_validator_passes_for_repo_curriculum() -> None:
    code, result = validate.validate()
    assert code == 0, result["errors"]
    assert result["summary"]["rows"] == 200
    assert result["summary"]["heldoutOverlapCount"] == 0


def test_manifest_preserves_claim_boundary() -> None:
    manifest = json.loads((ROOT / "training" / "tool_use" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "sophia.tool_use_curriculum.v1"
    assert manifest["candidateOnly"] is True
    assert manifest["canClaimAGI"] is False
    assert "canClaimAGI=false" in manifest["claimBoundary"]


def test_mcp_trace_schema_is_candidate_only() -> None:
    schema = json.loads((ROOT / "training" / "tool_use" / "mcp_trace_schema.json").read_text(encoding="utf-8"))
    assert schema["schema"] == "sophia.mcp_trace_row.v1"
    assert schema["candidateOnly"] is True
    assert schema["canClaimAGI"] is False
    assert {"messages", "toolTrace", "metadata"}.issubset(set(schema["requiredTopLevelFields"]))


def test_local_dataset_builder_knows_tool_use_dpo_pack() -> None:
    assert (
        "training/tool_use/dpo_pairs.jsonl",
        "dpo_tool_use_mcp.jsonl",
        "dpo",
    ) in build.DPO_SOURCES


def main() -> int:
    test_validator_passes_for_repo_curriculum()
    test_manifest_preserves_claim_boundary()
    test_mcp_trace_schema_is_candidate_only()
    test_local_dataset_builder_knows_tool_use_dpo_pack()
    print("test_tool_use_curriculum: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

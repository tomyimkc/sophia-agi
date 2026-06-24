#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the read-only OpenClaw MCP tool (sophia_mcp). All offline.

The real ``openclaw`` CLI is never invoked: ``tools_impl.subprocess.run`` is stubbed.
Audit-log writes go to a tempfile (mirrors test_mcp_audit.py), never the repo log.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_mcp import audit, tools_impl  # noqa: E402


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_run(fn):
    original = tools_impl.subprocess.run
    tools_impl.subprocess.run = fn
    return original


def test_infer_parses_json() -> None:
    payload = {"ok": True, "provider": "xai", "outputs": [{"text": "hello world"}]}
    original = _patch_run(lambda *a, **k: _FakeProc(0, json.dumps(payload)))
    try:
        out = tools_impl._openclaw_infer("xai/grok-4.3", "hi")
    finally:
        tools_impl.subprocess.run = original
    assert out["ok"] is True and out["text"] == "hello world" and out["provider"] == "xai"


def test_infer_binary_missing_degrades() -> None:
    def missing(*a, **k):
        raise FileNotFoundError("openclaw")

    original = _patch_run(missing)
    try:
        out = tools_impl._openclaw_infer("xai/grok-4.3", "hi")
    finally:
        tools_impl.subprocess.run = original
    assert out["ok"] is False and "FileNotFoundError" in (out["error"] or "")


def test_infer_empty_prompt_rejected() -> None:
    out = tools_impl._openclaw_infer("xai/grok-4.3", "")
    assert out["ok"] is False and "prompt" in (out["error"] or "")


def test_infer_is_low_risk_no_approval_needed() -> None:
    # risk="low" is registered by the @audited decorator at import, so it needs no approval
    os.environ.pop(audit.APPROVE_ENV, None)
    assert audit.TOOL_RISK.get("sophia_openclaw_infer") == "low"
    allowed, reason = audit.check_permission("sophia_openclaw_infer")
    assert allowed is True and reason is None


def test_infer_is_audited_to_temp_path() -> None:
    # mirror test_mcp_audit: re-decorate with a temp audit path so the run is logged ok
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "audit.jsonl"

        @audit.audited("sophia_openclaw_infer", risk="low", audit_path=log)
        def _tool():
            return {"ok": True, "text": "x"}

        out = _tool()
        assert out["ok"] is True
        records = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        assert records and records[-1]["tool"] == "sophia_openclaw_infer" and records[-1]["ok"] is True


def main() -> int:
    test_infer_parses_json()
    test_infer_binary_missing_degrades()
    test_infer_empty_prompt_rejected()
    test_infer_is_low_risk_no_approval_needed()
    test_infer_is_audited_to_temp_path()
    print("test_mcp_openclaw: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

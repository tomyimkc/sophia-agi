#!/usr/bin/env python3
"""Tests for the MCP audit + permission substrate."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_mcp import audit  # noqa: E402


def test_low_risk_allowed_and_logged() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "audit.jsonl"

        @audit.audited("demo_read", risk="low", audit_path=log)
        def read_tool(x):
            return {"value": x}

        out = read_tool(42)
        assert out == {"value": 42}
        lines = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        assert lines and lines[-1]["tool"] == "demo_read" and lines[-1]["ok"] is True


def test_medium_risk_blocked_without_approval() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "audit.jsonl"
        os.environ.pop(audit.APPROVE_ENV, None)

        @audit.audited("demo_write", risk="medium", audit_path=log)
        def write_tool():
            return {"ok": True}

        denied = write_tool()
        assert denied.get("denied") is True
        # approval flips it
        os.environ[audit.APPROVE_ENV] = "1"
        try:
            allowed = write_tool()
        finally:
            os.environ.pop(audit.APPROVE_ENV, None)
        assert allowed == {"ok": True}
        records = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        assert any(r.get("denied") for r in records)
        assert any(r.get("ok") and not r.get("denied") for r in records)


def test_exception_is_audited_not_raised() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "audit.jsonl"

        @audit.audited("demo_boom", risk="low", audit_path=log)
        def boom():
            raise ValueError("kaboom")

        out = boom()
        assert "error" in out and "kaboom" in out["error"]


def main() -> int:
    test_low_risk_allowed_and_logged()
    test_medium_risk_blocked_without_approval()
    test_exception_is_audited_not_raised()
    print("test_mcp_audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

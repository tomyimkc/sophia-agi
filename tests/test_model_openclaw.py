#!/usr/bin/env python3
"""Tests for the OpenClaw model provider (agent/model.py:_call_openclaw). All offline.

The real ``openclaw`` CLI is never invoked: ``m.subprocess.run`` is monkeypatched, so
these run with no daemon, no network, and no credentials (mirrors the urlopen stubbing
in test_model_adapter.py).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import model as m  # noqa: E402


class _FakeProc:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_run(fn):
    original = m.subprocess.run
    m.subprocess.run = fn
    return original


def test_preset_and_route_override() -> None:
    cfg = m.resolve_config("openclaw")
    assert cfg.kind == "openclaw" and cfg.model == "xai/grok-4.3"
    # the openclaw:<provider>/<model> route flows through as data (no resolve_config change)
    routed = m.resolve_config("openclaw:openai/gpt-4o")
    assert routed.kind == "openclaw" and routed.model == "openai/gpt-4o"


def test_openclaw_parses_json() -> None:
    payload = {"ok": True, "provider": "xai", "model": "grok-4.3", "outputs": [{"text": "Decision: yes."}]}
    original = _patch_run(lambda *a, **k: _FakeProc(0, json.dumps(payload)))
    try:
        result = m.ModelClient(m.resolve_config("openclaw")).generate("s", "u")
    finally:
        m.subprocess.run = original
    assert result.ok is True
    assert result.provider == "openclaw" and result.model == "xai/grok-4.3"
    assert "Decision" in result.text


def test_openclaw_nonzero_returncode_is_failure() -> None:
    original = _patch_run(lambda *a, **k: _FakeProc(1, "", "auth profile not found"))
    try:
        result = m.ModelClient(m.resolve_config("openclaw"), retries=1).generate("s", "u", sleep=lambda _: None)
    finally:
        m.subprocess.run = original
    assert result.ok is False
    assert "auth profile not found" in (result.error or "")


def test_openclaw_binary_missing_degrades() -> None:
    def missing(*a, **k):
        raise FileNotFoundError("openclaw")

    original = _patch_run(missing)
    try:
        result = m.ModelClient(m.resolve_config("openclaw"), retries=1).generate("s", "u", sleep=lambda _: None)
    finally:
        m.subprocess.run = original
    assert result.ok is False  # clean failure, never a crash
    assert "FileNotFoundError" in (result.error or "")


def test_openclaw_bad_json_degrades() -> None:
    original = _patch_run(lambda *a, **k: _FakeProc(0, "not json at all"))
    try:
        result = m.ModelClient(m.resolve_config("openclaw"), retries=1).generate("s", "u", sleep=lambda _: None)
    finally:
        m.subprocess.run = original
    assert result.ok is False


def test_openclaw_empty_outputs_is_not_ok() -> None:
    original = _patch_run(lambda *a, **k: _FakeProc(0, json.dumps({"ok": True, "outputs": []})))
    try:
        result = m.ModelClient(m.resolve_config("openclaw"), retries=1).generate("s", "u", sleep=lambda _: None)
    finally:
        m.subprocess.run = original
    assert result.ok is False


def test_openclaw_fallback_to_mock() -> None:
    os.environ.pop("SOPHIA_MOCK_RESPONSE", None)

    def missing(*a, **k):
        raise FileNotFoundError("openclaw")

    original = _patch_run(missing)
    try:
        client = m.ModelClient(m.resolve_config("openclaw"), [m.resolve_config("mock")], retries=1)
        result = client.generate("s", "u", sleep=lambda _: None)
    finally:
        m.subprocess.run = original
    assert result.ok is True
    assert result.provider == "mock"
    assert result.fallback_used is True


def main() -> int:
    test_preset_and_route_override()
    test_openclaw_parses_json()
    test_openclaw_nonzero_returncode_is_failure()
    test_openclaw_binary_missing_degrades()
    test_openclaw_bad_json_degrades()
    test_openclaw_empty_outputs_is_not_ok()
    test_openclaw_fallback_to_mock()
    print("test_model_openclaw: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

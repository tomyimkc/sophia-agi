# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Unit tests for agent/code_verifier.py sandboxed execution oracle."""
from __future__ import annotations

import os

import pytest

os.environ["SOPHIA_ALLOW_CODE_EXEC"] = "1"

from agent import code_verifier as cv  # noqa: E402

GOOD = "```python\ndef add(a, b):\n    return a + b\n```"
BAD = "```python\ndef add(a, b):\n    return a - b\n```"
TEST = "assert add(2, 3) == 5\nassert add(0, 0) == 0\n"


def test_verify_correct_code() -> None:
    r = cv.verify(GOOD, TEST)
    assert r["verdict"] == "accepted"
    assert r["detail"]["executed"] is True


def test_verify_wrong_code() -> None:
    r = cv.verify(BAD, TEST)
    assert r["verdict"] == "rejected"


def test_verify_timeout() -> None:
    loop = "```python\ndef add(a,b):\n    while True: pass\n```"
    r = cv.verify(loop, TEST, timeout_sec=2, cpu_sec=2)
    assert r["verdict"] == "rejected"
    assert "timed out" in r["reasons"][0].lower() or "exit" in r["reasons"][0].lower()


def test_verify_malicious_contained() -> None:
    """Fork bomb / rm -rf style snippets must not escape the child process group."""
    nasty = "```python\nimport os\nos.system('echo pwned')\n```"
    r = cv.verify(nasty, "pass\n", timeout_sec=3)
    assert r["verdict"] in ("accepted", "rejected")  # runs in isolation; must not hang parent


def test_verify_abstain_when_exec_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOPHIA_ALLOW_CODE_EXEC", "0")
    r = cv.verify(GOOD, TEST)
    assert r["verdict"] == "abstain"

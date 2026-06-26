#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the fail-closed kvcache client (agent.kvcache_client).

These run without a live server: they cover the contract that matters for
correctness — the cache is opt-in (disabled => from_env() is None) and a dead or
unreachable cache degrades to a miss/no-op rather than raising. A full
protocol-interop test against the Rust ``kvcache-server`` lives in the storage
crate's integration suite; here we only guarantee Sophia never breaks when the
cache is absent or down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import kvcache_client as kc  # noqa: E402


def test_from_env_disabled_when_unset(monkeypatch):
    monkeypatch.delenv("SOPHIA_KVCACHE_ADDR", raising=False)
    assert kc.from_env() is None


def test_from_env_malformed_returns_none(monkeypatch):
    monkeypatch.setenv("SOPHIA_KVCACHE_ADDR", "not-a-host-port")
    # Malformed config disables the cache rather than crashing the caller.
    assert kc.from_env() is None


def test_bad_address_validation():
    with pytest.raises(ValueError):
        kc.KVCacheClient("missing-port")


def test_lenient_ops_never_raise_when_server_down():
    # 127.0.0.1:1 is reserved/closed; lenient API must swallow the failure.
    client = kc.KVCacheClient("127.0.0.1:1", timeout=0.2)
    assert client.get(b"key") is None
    assert client.set(b"key", b"val") is False


def test_strict_ops_raise_when_server_down():
    client = kc.KVCacheClient("127.0.0.1:1", timeout=0.2)
    with pytest.raises((OSError, kc.KVCacheError)):
        client.get_strict(b"key")

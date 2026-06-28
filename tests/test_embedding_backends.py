# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the pluggable embedding-backend registry (the multilingual/multimodal seam)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import embedding_backends as eb  # noqa: E402


def test_builtins_registered() -> None:
    avail = eb.available()
    assert "local-hash-v1" in avail
    assert "gemini" in avail


def test_local_hash_backend_resolves_and_embeds() -> None:
    fn = eb.get("local-hash-v1")
    assert fn is not None
    vec = fn("who wrote the dao de jing")
    assert len(vec) > 0


def test_unknown_backend_returns_none() -> None:
    assert eb.get("does-not-exist") is None


def test_register_and_resolve_custom_backend() -> None:
    # Simulates registering a learned multilingual/multimodal embedder under a new id.
    sentinel = object()
    eb.register("test-fake-v1", lambda: (lambda _text: sentinel))
    try:
        fn = eb.get("test-fake-v1")
        assert fn is not None and fn("任何语言") is sentinel
        assert "test-fake-v1" in eb.available()
    finally:
        eb._REGISTRY.pop("test-fake-v1", None)


def test_loader_failure_is_swallowed() -> None:
    def _boom():
        raise RuntimeError("missing dependency")

    eb.register("test-boom", _boom)
    try:
        assert eb.get("test-boom") is None  # loader raised → None, not a crash
    finally:
        eb._REGISTRY.pop("test-boom", None)

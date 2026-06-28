# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Pluggable embedding-backend registry — the extension point for new embedders.

Retrieval embeds a query with the SAME backend that built the index (so query and document
vectors share one space). Today two backends ship: the offline ``local-hash-v1`` hashing
embedder and the API-backed ``gemini``. This registry is the seam where **new** embedders —
crucially a learned **multilingual** or **multimodal** model, the JD's "任何语言、任何模态都能被
平等地检索" — register *without editing the retrieval path*: build the index under a new backend
id, register a query-embed function under that id here, and `agent.retrieval.embed_query_for_index`
picks it up automatically.

A backend is just ``backend_id -> (text -> np.ndarray)``. Registration is lazy (a thunk that
imports on first use) so adding a heavy backend never imposes its import cost — or its
dependency — on the offline default path. Shipping the actual learned weights/model is
deliberately out of scope here (it would break the airgap, no-dependency CI); this module is
the contract those weights plug into.
"""

from __future__ import annotations

from typing import Callable

#: backend_id -> thunk returning an ``embed_query(text) -> np.ndarray`` callable.
_REGISTRY: "dict[str, Callable[[], Callable[[str], object]]]" = {}


def register(backend_id: str, loader: "Callable[[], Callable[[str], object]]") -> None:
    """Register a backend by id. ``loader`` is a thunk returning the embed_query callable.

    Lazy by contract: ``loader`` is only called when a query actually needs this backend, so a
    heavy/optional dependency stays unimported until used.
    """
    _REGISTRY[backend_id] = loader


def get(backend_id: str) -> "Callable[[str], object] | None":
    """Resolve a registered backend's embed_query callable, or ``None`` if absent/unloadable."""
    loader = _REGISTRY.get(backend_id)
    if loader is None:
        return None
    try:
        return loader()
    except Exception:
        return None


def available() -> "list[str]":
    """Registered backend ids (sorted)."""
    return sorted(_REGISTRY)


def _load_local_hash():
    from agent.rag_local_embed import embed_query

    return embed_query


def _load_gemini():
    from agent.rag_embed import embed_query

    return embed_query


# Built-ins. The retrieval path still has fast inline branches for these two; registering them
# here keeps the registry a complete, truthful picture of what backends exist and lets generic
# tooling enumerate them.
register("local-hash-v1", _load_local_hash)
register("gemini", _load_gemini)


__all__ = ["available", "get", "register"]

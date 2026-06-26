# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the opt-in learned embedders + their graceful absence in the registry.

CI-safe: sentence-transformers is an optional dependency, so the absence path (registry returns
None → retrieval falls back to the committed backend) is the invariant tested everywhere; a live
cross-lingual round-trip runs only when the dependency happens to be installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from agent import embedding_backends as eb  # noqa: E402
from agent import embedding_st as est  # noqa: E402


def test_models_registry_shape() -> None:
    assert set(est.MODELS) == {"st-multilingual-v1", "clip-multimodal-v1"}
    name, dim, modality = est.MODELS["st-multilingual-v1"]
    assert dim == 384 and modality == "text"
    assert est.MODELS["clip-multimodal-v1"][2] == "text+image"


def test_learned_backends_are_registered() -> None:
    avail = eb.available()
    assert "st-multilingual-v1" in avail
    assert "clip-multimodal-v1" in avail


def test_registry_get_matches_dependency_presence() -> None:
    fn = eb.get("st-multilingual-v1")
    if est.is_available():
        assert callable(fn)
    else:
        # Graceful absence: loader probes the dep and raises → get() returns None,
        # so retrieval falls back to the committed offline backend.
        assert fn is None


def test_embed_image_rejects_non_multimodal_backend_without_model() -> None:
    # The modality guard fires before any model load, so this is offline-safe.
    with pytest.raises(ValueError):
        est.embed_image("x.png", backend_id="st-multilingual-v1")


def test_make_query_embedder_is_callable() -> None:
    assert callable(est.make_query_embedder("st-multilingual-v1"))


@pytest.mark.skipif(not est.is_available(), reason="sentence-transformers not installed")
def test_live_cross_lingual_recall() -> None:
    import numpy as np

    # A Chinese query should be closer to its English match than to an unrelated English line.
    q = est.embed_query("谁写了道德经？")
    match = est.embed_query("The Dao De Jing is attributed to Laozi.")
    other = est.embed_query("Photosynthesis converts light into chemical energy.")
    assert float(np.dot(q, match)) > float(np.dot(q, other))

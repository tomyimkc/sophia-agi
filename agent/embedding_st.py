# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Learned multilingual + multimodal embedders (opt-in) — the registry's quality lever.

The committed default embedder (`agent/rag_local_embed.py`, ``local-hash-v1``) is a *lexical*
hash: offline, deterministic, airgap-safe, but it generalizes over surface form, not deep
meaning. This module is the realized answer to the original brief — "任何语言、任何模态都能被平等地
检索" — via real **learned** models, plugged in through the embedder registry
(`agent/embedding_backends.py`) so retrieval picks them up with no change to the retrieval path:

  - ``st-multilingual-v1`` — a multilingual sentence encoder (paraphrase-multilingual-MiniLM):
    cross-lingual *meaning*, so a Chinese query can recall an English passage and vice-versa;
  - ``clip-multimodal-v1`` — a CLIP text+image encoder: text and images in ONE space, so an
    image can be retrieved by a text query and vice-versa (the "any modality" half).

Deliberately **not** the committed default. A learned model needs weights (a network download)
and produces platform-sensitive floats — both of which would break the repo's airgap +
deterministic-CI + reproducible-index guarantees, and committing a multi-MB learned index is its
own decision. So this is **opt-in**: it requires ``pip install sentence-transformers`` (+ Pillow
for images) and a one-time model fetch; absent those, the registry loader fails gracefully and
retrieval keeps using the committed offline backend. Build a learned index with
``tools/build_rag_index.py --st st-multilingual-v1`` (operator-run). Vectors are L2-normalized
so the existing cosine search (`agent/vector_store.py`) works unchanged.
"""

from __future__ import annotations

from pathlib import Path

#: backend_id -> (model name, embedding dim, modality). Dims documented for the index manifest.
MODELS: "dict[str, tuple[str, int, str]]" = {
    "st-multilingual-v1": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 384, "text"),
    "clip-multimodal-v1": ("sentence-transformers/clip-ViT-B-32", 512, "text+image"),
}

_MODEL_CACHE: dict = {}


def is_available() -> bool:
    """True iff the optional ``sentence-transformers`` dependency can be imported."""
    try:
        import sentence_transformers  # noqa: F401

        return True
    except Exception:
        return False


def _load_model(backend_id: str):
    if backend_id not in MODELS:
        raise KeyError(f"unknown learned backend {backend_id!r} (have: {', '.join(MODELS)})")
    model_name = MODELS[backend_id][0]
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer  # optional dependency

        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def _as_f32_list(matrix):
    import numpy as np

    return [np.asarray(row, dtype=np.float32) for row in matrix]


def embed_texts(texts: "list[str]", *, backend_id: str = "st-multilingual-v1") -> "list":
    """Embed texts into L2-normalized float32 vectors with the given learned backend."""
    model = _load_model(backend_id)
    matrix = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return _as_f32_list(matrix)


def embed_query(text: str, *, backend_id: str = "st-multilingual-v1"):
    """Embed a single query string (same space as :func:`embed_texts`)."""
    return embed_texts([text], backend_id=backend_id)[0]


def embed_image(image, *, backend_id: str = "clip-multimodal-v1"):
    """Embed an image (path or PIL.Image) into the SAME space as text for a CLIP backend.

    Enables cross-modal recall: an image indexed here is retrievable by a text query embedded
    with the same backend. Requires Pillow.
    """
    if MODELS[backend_id][2] != "text+image":
        raise ValueError(f"backend {backend_id!r} is not multimodal")
    model = _load_model(backend_id)
    if isinstance(image, (str, Path)):
        from PIL import Image  # optional dependency

        image = Image.open(image)
    matrix = model.encode([image], convert_to_numpy=True, normalize_embeddings=True)
    return _as_f32_list(matrix)[0]


def make_query_embedder(backend_id: str):
    """Return an ``embed_query(text) -> vec`` closure for the registry (matches its contract)."""
    def _embed(text: str):
        return embed_query(text, backend_id=backend_id)

    return _embed


__all__ = [
    "MODELS", "embed_image", "embed_query", "embed_texts", "is_available", "make_query_embedder",
]

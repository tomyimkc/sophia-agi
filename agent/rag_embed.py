"""Gemini / Vertex embedding helpers."""

from __future__ import annotations

import time

import numpy as np

from agent.google_genai_client import gemini_embed_model, make_genai_client


def embed_texts(texts: list[str], *, batch_size: int = 16) -> list[np.ndarray]:
    if not texts:
        return []
    client = make_genai_client()
    model = gemini_embed_model()
    vectors: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.models.embed_content(model=model, contents=batch)
        for item in response.embeddings:
            values = getattr(item, "values", None) or item
            vectors.append(np.asarray(values, dtype=np.float32))
        time.sleep(0.05)

    return vectors


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
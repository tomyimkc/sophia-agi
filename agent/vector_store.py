# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Local vector index for Sophia RAG."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent.config import ROOT
from agent.retrieval import SourceChunk, _score, _tokenize

DEFAULT_INDEX_DIR = ROOT / "rag" / "index"
CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npz"

_LOG = logging.getLogger("sophia.rag")
_warned_no_embeddings: set[str] = set()


@dataclass
class IndexedChunk:
    chunk_id: str
    path: str
    title: str
    text: str
    domain: str | None
    kind: str
    embedding: np.ndarray | None = None

    @classmethod
    def from_row(cls, row: dict, embedding: np.ndarray | None = None) -> "IndexedChunk":
        return cls(
            chunk_id=row["chunkId"],
            path=row["path"],
            title=row["title"],
            text=row["text"],
            domain=row.get("domain"),
            kind=row.get("kind", "source"),
            embedding=embedding,
        )


def index_dir(path: Path | None = None) -> Path:
    import os

    if path is not None:
        return path
    env = os.environ.get("SOPHIA_RAG_INDEX_DIR", "").strip()
    if env:
        return Path(env)
    return DEFAULT_INDEX_DIR


def save_index(chunks: list[IndexedChunk], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    embeddings: list[np.ndarray] = []
    has_embed = False
    for chunk in chunks:
        row = {
            "chunkId": chunk.chunk_id,
            "path": chunk.path,
            "title": chunk.title,
            "text": chunk.text,
            "domain": chunk.domain,
            "kind": chunk.kind,
        }
        rows.append(row)
        if chunk.embedding is not None:
            embeddings.append(np.asarray(chunk.embedding, dtype=np.float32))
            has_embed = True

    (out_dir / CHUNKS_FILE).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    if has_embed and embeddings:
        matrix = np.vstack(embeddings)
        np.savez_compressed(out_dir / EMBEDDINGS_FILE, embeddings=matrix)


def load_index(index_path: Path | None = None) -> list[IndexedChunk]:
    root = index_path or index_dir()
    chunks_path = root / CHUNKS_FILE
    if not chunks_path.exists():
        return []

    rows = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    embed_path = root / EMBEDDINGS_FILE
    matrix = None
    if embed_path.exists():
        matrix = np.load(embed_path)["embeddings"]
    elif rows:
        # Non-silent fallback: chunks exist but there are no embeddings, so search()
        # will degrade to keyword-only scoring. Warn once per index dir so this
        # capability loss is visible rather than silent (see RESULTS.md retrieval notes).
        key = str(root)
        if key not in _warned_no_embeddings:
            _warned_no_embeddings.add(key)
            _LOG.warning(
                "RAG index at %s has no %s — semantic search UNAVAILABLE; retrieval is "
                "keyword-only. Build embeddings with `python tools/build_rag_index.py "
                "--embed` (requires GOOGLE_API_KEY or Vertex).",
                root,
                EMBEDDINGS_FILE,
            )

    loaded: list[IndexedChunk] = []
    for i, row in enumerate(rows):
        emb = matrix[i] if matrix is not None and i < len(matrix) else None
        loaded.append(IndexedChunk.from_row(row, emb))
    return loaded


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / denom)


def search(
    query: str,
    chunks: list[IndexedChunk],
    *,
    top_k: int = 8,
    query_embedding: np.ndarray | None = None,
) -> list[SourceChunk]:
    if not chunks:
        return []

    if query_embedding is not None and chunks and chunks[0].embedding is not None:
        scored = []
        for chunk in chunks:
            if chunk.embedding is None:
                continue
            score = _cosine(query_embedding, chunk.embedding)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SourceChunk(
                path=item.path,
                title=item.title,
                excerpt=item.text[:1200] + ("..." if len(item.text) > 1200 else ""),
                score=score,
            )
            for score, item in scored[:top_k]
        ]

    query_tokens = _tokenize(query)
    ranked: list[SourceChunk] = []
    for chunk in chunks:
        score = _score(query_tokens, f"{chunk.title} {chunk.text}")
        if score <= 0:
            continue
        ranked.append(
            SourceChunk(
                path=chunk.path,
                title=chunk.title,
                excerpt=chunk.text[:1200] + ("..." if len(chunk.text) > 1200 else ""),
                score=score,
            )
        )
    ranked.sort(key=lambda c: c.score, reverse=True)
    return ranked[:top_k]
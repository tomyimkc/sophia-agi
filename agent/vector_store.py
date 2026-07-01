# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Local vector index for Sophia RAG."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from agent.config import ROOT
from agent.retrieval import SourceChunk, _score, _tokenize

DEFAULT_INDEX_DIR = ROOT / "rag" / "index"
CHUNKS_FILE = "chunks.jsonl"
EMBEDDINGS_FILE = "embeddings.npz"
META_FILE = "embeddings.meta.json"

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


def embedding_backend_id(index_path: Path | None = None) -> "str | None":
    """Backend id stamped into the index manifest (e.g. ``local-hash-v1``, ``gemini``).

    Lets retrieval embed the query with the SAME backend that built the index instead of
    guessing from env. Returns None when no manifest is present (legacy / chunks-only index).
    """
    meta_path = (index_path or index_dir()) / META_FILE
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")).get("backend")
    except (ValueError, OSError):
        return None


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


# --- optional kvcache acceleration (storage/kvcache) -----------------------
# Opt-in via SOPHIA_KVCACHE_ADDR. Caches search *results* keyed by the query,
# top_k, and a fingerprint of the chunk set + query embedding, so a repeated
# query against an unchanged index skips re-scoring. Fail-closed: any cache
# error degrades to a normal (uncached) search — correctness never depends on it.

_KVCACHE_TTL_MS = int(os.environ.get("SOPHIA_KVCACHE_TTL_MS", "300000"))


def _chunkset_fingerprint(chunks: list[IndexedChunk]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(str(len(chunks)).encode())
    # First/last/middle chunk ids cheaply detect an index swap without hashing all rows.
    for c in (chunks[0], chunks[len(chunks) // 2], chunks[-1]):
        h.update(b"\x00")
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x01")
        h.update(b"1" if c.embedding is not None else b"0")
    return h.hexdigest()


def _cache_key(query: str, top_k: int, chunks: list[IndexedChunk], query_embedding) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    h.update(query.encode("utf-8"))
    h.update(f"|k={top_k}|".encode())
    h.update(_chunkset_fingerprint(chunks).encode())
    if query_embedding is not None:
        h.update(np.asarray(query_embedding, dtype=np.float32).tobytes())
    return b"sophia:rag:search:" + h.hexdigest().encode()


def _encode_results(results: list[SourceChunk]) -> bytes:
    return json.dumps([asdict(r) for r in results], ensure_ascii=False).encode("utf-8")


def _decode_results(blob: bytes) -> list[SourceChunk]:
    return [SourceChunk(**row) for row in json.loads(blob.decode("utf-8"))]


def search(
    query: str,
    chunks: list[IndexedChunk],
    *,
    top_k: int = 8,
    query_embedding: np.ndarray | None = None,
) -> list[SourceChunk]:
    if not chunks:
        return []

    client = None
    key = None
    try:
        from agent import kvcache_client

        client = kvcache_client.from_env()
        if client is not None:
            key = _cache_key(query, top_k, chunks, query_embedding)
            hit = client.get(key)
            if hit is not None:
                try:
                    return _decode_results(hit)
                except (ValueError, TypeError) as e:  # poisoned/stale entry
                    _LOG.debug("kvcache hit failed to decode, recomputing: %s", e)
    except Exception as e:  # import or key-build failure must never break search
        _LOG.debug("kvcache lookup skipped: %s", e)
        client = None

    results = _search_impl(query, chunks, top_k=top_k, query_embedding=query_embedding)

    if client is not None and key is not None:
        with contextlib.closing(client):
            try:
                client.set(key, _encode_results(results), _KVCACHE_TTL_MS)
            except Exception as e:  # noqa: BLE001 — cache write is best-effort
                _LOG.debug("kvcache store skipped: %s", e)
    return results


def _search_impl(
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
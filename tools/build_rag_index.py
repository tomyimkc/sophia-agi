#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build curated Sophia RAG index (no benchmark holdouts).

Usage:
  python tools/build_rag_index.py --dry-run
  python tools/build_rag_index.py                 # chunks only (no embeddings)
  python tools/build_rag_index.py --local         # + offline hashing embeddings (airgap-safe)
  python tools/build_rag_index.py --embed         # + Gemini/Vertex embeddings (needs a key)
  python tools/build_rag_index.py --verify        # rebuild & assert committed manifest hash

The ``--local`` backend (``agent.rag_local_embed``) is deterministic, CPU-only, and needs no
API key, so it makes vector recall live under ``SOPHIA_PROFILE=airgap`` — and reproducibly:
``--verify`` rebuilds the embeddings in-memory and checks their sha256 against the committed
``embeddings.meta.json``. The manifest records which backend produced the vectors so
``agent.retrieval.retrieve`` embeds queries with the matching embedder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag_sources import collect_curated_chunks  # noqa: E402
from agent.vector_store import (  # noqa: E402
    DEFAULT_INDEX_DIR, EMBEDDINGS_FILE, IndexedChunk, save_index,
)

META_FILE = "embeddings.meta.json"


def _embed(indexed: "list[IndexedChunk]", backend: str) -> str:
    """Attach embeddings to ``indexed`` in place; return the backend id stamped into the index."""
    texts = [f"{c.title}\n{c.text}" for c in indexed]
    if backend == "local":
        from agent.rag_local_embed import BACKEND_ID, embed_texts
        backend_id = BACKEND_ID
    elif backend == "gemini":
        from agent.rag_embed import embed_texts
        backend_id = "gemini"
    else:
        raise SystemExit(f"unknown backend {backend!r}")
    vectors = embed_texts(texts)
    for chunk, vector in zip(indexed, vectors, strict=True):
        chunk.embedding = vector
    return backend_id


def _manifest(indexed: "list[IndexedChunk]", backend_id: str) -> dict:
    matrix = np.vstack([np.asarray(c.embedding, dtype=np.float32) for c in indexed])
    payload = "\n".join(c.chunk_id for c in indexed).encode("utf-8")
    # Hash a QUANTIZED (1e-6) int view, not raw float bytes, so the reproducibility check
    # tolerates last-ULP numerical noise across numpy versions / platforms while still
    # catching any real change to the vectors.
    quantized = np.rint(matrix * 1_000_000.0).astype(np.int64)
    return {
        "backend": backend_id,
        "dim": int(matrix.shape[1]),
        "count": int(matrix.shape[0]),
        "chunkIdsSha256": hashlib.sha256(payload).hexdigest(),
        "embeddingsSha256": hashlib.sha256(np.ascontiguousarray(quantized).tobytes()).hexdigest(),
    }


def _to_indexed(raw) -> "list[IndexedChunk]":
    return [IndexedChunk(chunk_id=c.chunk_id, path=c.path, title=c.title,
                         text=c.text, domain=c.domain, kind=c.kind) for c in raw]


def verify(out_dir: Path) -> int:
    meta_path = out_dir / META_FILE
    if not meta_path.exists():
        print(f"FAIL: no {META_FILE} at {out_dir} to verify against", file=sys.stderr)
        return 1
    committed = json.loads(meta_path.read_text(encoding="utf-8"))
    backend = "gemini" if committed.get("backend") == "gemini" else "local"
    if backend == "gemini":
        print("SKIP: gemini-built index is not deterministically reproducible offline")
        return 0
    indexed = _to_indexed(collect_curated_chunks())
    _embed(indexed, backend)
    rebuilt = _manifest(indexed, committed.get("backend"))
    ok = (rebuilt["embeddingsSha256"] == committed.get("embeddingsSha256")
          and rebuilt["chunkIdsSha256"] == committed.get("chunkIdsSha256"))
    if ok:
        print(f"OK: reproducible — backend={rebuilt['backend']} dim={rebuilt['dim']} "
              f"count={rebuilt['count']} sha={rebuilt['embeddingsSha256'][:12]}")
        return 0
    print("FAIL: rebuilt index does not match committed manifest", file=sys.stderr)
    print(f"  chunkIds:   {rebuilt['chunkIdsSha256'][:12]} vs {committed.get('chunkIdsSha256','')[:12]}",
          file=sys.stderr)
    print(f"  embeddings: {rebuilt['embeddingsSha256'][:12]} vs {committed.get('embeddingsSha256','')[:12]}",
          file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Build curated Sophia RAG index")
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--embed", action="store_true", help="Compute Gemini/Vertex embeddings")
    parser.add_argument("--local", action="store_true",
                        help="Compute offline deterministic hashing embeddings (airgap-safe)")
    parser.add_argument("--verify", action="store_true",
                        help="Rebuild in-memory and assert the committed manifest hash matches")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.verify:
        return verify(args.out)
    if args.embed and args.local:
        raise SystemExit("choose one backend: --local or --embed (not both)")

    raw = collect_curated_chunks()
    print(f"Chunks: {len(raw)}")
    kinds: dict[str, int] = {}
    for chunk in raw:
        kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1
    print("Kinds:", kinds)

    if args.dry_run:
        return 0

    indexed = _to_indexed(raw)

    backend = "gemini" if args.embed else ("local" if args.local else None)
    if backend:
        print(f"Embedding with {backend} backend...")
        backend_id = _embed(indexed, backend)

    save_index(indexed, args.out)

    if backend:
        manifest = _manifest(indexed, backend_id)
        (args.out / META_FILE).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.out / EMBEDDINGS_FILE} + {META_FILE}: {json.dumps(manifest)}")
    else:
        print(f"Wrote {args.out} (chunks only — no embeddings; pass --local or --embed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build curated Sophia RAG index (no benchmark holdouts).

Usage:
  python tools/build_rag_index.py --dry-run
  python tools/build_rag_index.py
  python tools/build_rag_index.py --embed   # requires GOOGLE_API_KEY or Vertex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag_embed import embed_texts  # noqa: E402
from agent.rag_sources import collect_curated_chunks  # noqa: E402
from agent.vector_store import DEFAULT_INDEX_DIR, IndexedChunk, save_index  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build curated Sophia RAG index")
    parser.add_argument("--out", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--embed", action="store_true", help="Compute Gemini/Vertex embeddings")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    raw = collect_curated_chunks()
    print(f"Chunks: {len(raw)}")
    kinds: dict[str, int] = {}
    for chunk in raw:
        kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1
    print("Kinds:", kinds)

    if args.dry_run:
        return 0

    indexed: list[IndexedChunk] = [
        IndexedChunk(
            chunk_id=c.chunk_id,
            path=c.path,
            title=c.title,
            text=c.text,
            domain=c.domain,
            kind=c.kind,
        )
        for c in raw
    ]

    if args.embed:
        print("Embedding with Gemini/Vertex...")
        texts = [f"{c.title}\n{c.text}" for c in indexed]
        vectors = embed_texts(texts)
        for chunk, vector in zip(indexed, vectors, strict=True):
            chunk.embedding = vector

    save_index(indexed, args.out)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
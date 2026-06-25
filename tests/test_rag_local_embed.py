#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the offline local hashing embedding + live vector recall.

Covers: deterministic/reproducible vectors (stable across processes, L2-normalized);
cosine recall surfaces a semantically-overlapping chunk that the exact-token keyword
scorer misses; provenance ids survive into the search results; and the committed
rag/index manifest verifies (build is reproducible). Offline, numpy-only, no API key.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.rag_local_embed import BACKEND_ID, DIM, embed_query, embed_text  # noqa: E402
from agent.vector_store import IndexedChunk, search  # noqa: E402


def test_embedding_is_normalized_and_fixed_width() -> None:
    v = embed_text("Confucius and the Analects")
    assert v.shape == (DIM,) and v.dtype == np.float32
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_embedding_is_deterministic_across_processes() -> None:
    # blake2b (not the salted builtin hash) must give identical vectors in a fresh process.
    code = ("import numpy as np, sys; sys.path.insert(0, %r);"
            "from agent.rag_local_embed import embed_text;"
            "print(float(np.sum(embed_text('the Dao De Jing by Laozi'))))" % str(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    here = float(np.sum(embed_text("the Dao De Jing by Laozi")))
    assert abs(here - float(out.stdout.strip())) < 1e-6


def test_empty_text_is_zero_vector() -> None:
    assert float(np.linalg.norm(embed_text(""))) == 0.0


def _chunk(cid, title, text, **meta):
    c = IndexedChunk(chunk_id=cid, path=cid, title=title, text=text,
                     domain=meta.get("domain"), kind=meta.get("kind", "data"))
    c.embedding = embed_text(f"{title}\n{text}")
    return c


def test_vector_recall_beats_keyword_on_morphological_overlap() -> None:
    # Query shares NO exact 3+ char token with the relevant chunk's tokens by the keyword
    # tokenizer's lens, but char-trigram hashing captures the "calligraph" stem overlap.
    chunks = [
        _chunk("calligraphy", "Chinese calligraphy",
               "The art of calligraphic brushwork and ink writing traditions."),
        _chunk("astronomy", "Astronomy", "Stars, planets, orbital mechanics and telescopes."),
    ]
    q = "calligrapher"
    qv = embed_query(q)
    hits = search(q, chunks, top_k=2, query_embedding=qv)
    assert hits and hits[0].title == "Chinese calligraphy"
    # keyword-only path (no query_embedding) finds nothing for this non-overlapping token
    assert search(q, chunks, top_k=2) == []


def test_search_preserves_chunk_identity() -> None:
    chunks = [_chunk("analects", "Analects", "Confucius teachings compiled by disciples.")]
    hits = search("Confucius", chunks, top_k=1, query_embedding=embed_query("Confucius"))
    assert hits and hits[0].path == "analects" and hits[0].title == "Analects"


def test_committed_index_manifest_is_reproducible() -> None:
    # Only meaningful when the local index has been committed; skip cleanly otherwise.
    from agent.vector_store import META_FILE, embedding_backend_id, index_dir

    idir = index_dir()
    if not (idir / META_FILE).exists() or embedding_backend_id(idir) != BACKEND_ID:
        return  # no committed local index in this checkout — nothing to verify
    rc = subprocess.run([sys.executable, "tools/build_rag_index.py", "--verify"],
                        cwd=str(ROOT), capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr + rc.stdout


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("all passed")

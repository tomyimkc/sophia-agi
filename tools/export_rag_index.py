#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Export the committed RAG index to the text vectors format the Rust ANN server reads.

Writes one line per chunk — ``<row_index> f0 f1 …`` — so the dense embeddings can be served by
``services/ann_serving`` (`serve`). The id is the chunk's **row index** in
``agent.vector_store.load_index`` order, so a returned id maps straight back to the loaded
chunk on the Python side (`agent/ann_client.py`) with no extra sidecar.

  python tools/export_rag_index.py                 # -> rag/index/vectors.txt
  python tools/export_rag_index.py --out PATH

Offline; needs the committed ``embeddings.npz`` (build it with tools/build_rag_index.py --local).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "rag" / "index" / "vectors.txt"


def export(out_path: Path) -> int:
    """Write the vectors file; return the number of chunks exported."""
    from agent.vector_store import index_dir, load_index

    indexed = load_index(index_dir())
    rows = [(i, c) for i, c in enumerate(indexed) if c.embedding is not None]
    if not rows:
        raise SystemExit(
            "no embeddings in the index — build them with "
            "`python tools/build_rag_index.py --local` first."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, chunk in rows:
            # repr() round-trips the float exactly; Rust parses it as f32 (the embedder emits f32).
            vec = " ".join(repr(float(x)) for x in chunk.embedding)
            fh.write(f"{i} {vec}\n")
    return len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export RAG embeddings for the Rust ANN server")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    n = export(args.out)
    print(f"Exported {n} vectors -> {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

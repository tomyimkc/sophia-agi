# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tests for the Python↔Rust ANN bridge (export format + fail-soft client + round-trip).

The live round-trip is gated on the Rust `serve` binary being built (it isn't in the
pure-Python CI job), so this suite stays green without a cargo toolchain.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from agent.ann_client import AnnClient  # noqa: E402


def test_export_writes_id_then_floats(tmp_path) -> None:
    from tools.export_rag_index import export

    out = tmp_path / "vectors.txt"
    n = export(out)
    assert n > 0
    first = out.read_text(encoding="utf-8").splitlines()[0].split()
    assert first[0] == "0"  # row index as id
    assert len(first) > 1  # followed by the vector
    float(first[1])  # parseable float


def test_client_reports_unavailable_when_binary_missing(tmp_path) -> None:
    # Point at non-existent paths → available() False and start() raises (no silent failure).
    client = AnnClient(vectors_path=tmp_path / "nope.txt", binary=tmp_path / "nobin")
    assert client.available() is False
    with pytest.raises(RuntimeError):
        client.start()


def test_live_roundtrip_matches_python_exact_when_built() -> None:
    client = AnnClient()
    if not client.available():
        pytest.skip("Rust serve binary or vectors.txt not present (cargo build + export needed)")

    import numpy as np

    from agent.rag_local_embed import embed_query
    from agent.vector_store import index_dir, load_index

    idx = load_index(index_dir())
    q = embed_query("who wrote the dao de jing")

    def cos(a, b) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))

    py_top1 = max(
        ((cos(q, c.embedding), i) for i, c in enumerate(idx) if c.embedding is not None)
    )[1]

    with client as cl:
        assert cl.size > 0 and cl.dim > 0
        hits = cl.search(q, k=5, ef=128)
    assert hits, "server returned no hits"
    # High ef → the graph should recover the exact nearest neighbour.
    assert hits[0][0] == py_top1


def test_sharded_serve_recovers_exact_top1_when_built() -> None:
    client = AnnClient(shards=4)
    if not client.available():
        pytest.skip("Rust serve binary or vectors.txt not present")
    from agent.rag_local_embed import embed_query
    from agent.vector_store import index_dir, load_index

    idx = load_index(index_dir())
    q = embed_query("who wrote the dao de jing")
    with client as cl:
        assert cl.size == sum(1 for c in idx if c.embedding is not None)
        hits = cl.search(q, k=5, ef=128)
    assert hits and 0 <= hits[0][0] < len(idx)


def test_pack_then_serve_from_idx_roundtrip(tmp_path) -> None:
    import subprocess

    from agent.ann_client import DEFAULT_BINARY, DEFAULT_VECTORS

    pack_bin = DEFAULT_BINARY.parent / "pack"
    if not (pack_bin.exists() and DEFAULT_VECTORS.exists()):
        pytest.skip("pack binary or vectors.txt not present (cargo build --release + export)")

    idx_path = tmp_path / "shard.idx"
    out = subprocess.run([str(pack_bin), str(DEFAULT_VECTORS), str(idx_path), "4", "16", "200"],
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and idx_path.exists(), out.stderr

    # Serving from the .idx must return results identical to building from text.
    from agent.rag_local_embed import embed_query

    q = embed_query("who wrote the dao de jing")
    text_client, idx_client = AnnClient(), AnnClient(vectors_path=idx_path)
    if not text_client.available():
        pytest.skip("serve binary or vectors.txt not present")
    with text_client as tc:
        from_text = tc.search(q, k=5, ef=128)
    with idx_client as ic:
        from_idx = ic.search(q, k=5, ef=128)
    assert from_idx == from_text  # persistence is lossless

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

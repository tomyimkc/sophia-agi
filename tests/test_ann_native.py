# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Gated test for the in-process PyO3 binding (the `python` cargo feature).

The native extension is built only with `cargo build --features python` (off by default), so
this test skips unless the `.so` is present — CI without the optional build stays green.
"""

from __future__ import annotations

import importlib.util
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


def _find_native() -> "Path | None":
    base = ROOT / "services" / "ann_serving" / "target"
    for sub in ("release", "debug"):
        cand = base / sub / "libsophia_ann.so"
        if cand.exists():
            return cand
    return None


def _load_native(tmp_path):
    so = _find_native()
    if so is None:
        pytest.skip("native sophia_ann not built (cargo build --release --features python)")
    dst = tmp_path / "sophia_ann.so"
    shutil.copy(so, dst)
    spec = importlib.util.spec_from_file_location("sophia_ann", dst)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def test_native_sharded_index_roundtrip(tmp_path) -> None:
    sa = _load_native(tmp_path)
    idx = sa.ShardedHnsw(num_shards=4, dim=4, m=16, ef_construction=64)
    for i in range(200):
        idx.add(i, _unit([i % 7, (i % 3) + 1, 2.0, (i % 5) + 1]))
    assert len(idx) == 200
    assert idx.dim == 4 and idx.num_shards == 4
    assert sum(idx.shard_sizes()) == 200

    hits = idx.search(_unit([1, 2, 2, 1]), k=5, ef=64)
    assert hits and len(hits) == 5

    # Persistence is lossless across the binding too.
    out = tmp_path / "native.idx"
    idx.save(str(out))
    reloaded = sa.ShardedHnsw.load(str(out))
    assert len(reloaded) == 200
    assert reloaded.search(_unit([1, 2, 2, 1]), k=5, ef=64) == hits


def test_native_rejects_wrong_dim(tmp_path) -> None:
    sa = _load_native(tmp_path)
    idx = sa.ShardedHnsw(num_shards=2, dim=4)
    with pytest.raises(ValueError):
        idx.add(0, [1.0, 2.0])  # wrong dimensionality
    with pytest.raises(ValueError):
        idx.search([1.0, 2.0], k=5, ef=16)

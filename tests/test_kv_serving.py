# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Offline invariants for the KV-cache serving layer (no torch, no GPU).

Mirrors the repo's RLVR-style discipline: the *policy* (paging, prefix sharing,
tiered eviction, cache-aware routing) is proven deterministically on any
machine. The GPU tensor movement is out of scope here and lives behind the
gated live paths described in docs/SYSTEMS-TRACK.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving import kv_cache, load_balancer  # noqa: E402
from serving.kv_cache import BlockTier, TieredKVCache, block_hashes  # noqa: E402
from serving.load_balancer import CacheAwareRouter  # noqa: E402


# ---- KV cache --------------------------------------------------------------

def test_kv_offline_invariants() -> None:
    ok, detail = kv_cache.offline_invariants()
    assert ok, detail["checks"]


def test_block_hashes_prefix_property() -> None:
    a = block_hashes([1, 2, 3, 4, 5, 6, 7, 8], 4)
    b = block_hashes([1, 2, 3, 4, 9, 9, 9, 9], 4)
    assert a[0] == b[0]      # shared prefix block
    assert a[1] != b[1]      # divergent block
    # trailing partial block is not sealed
    assert block_hashes([1, 2, 3, 4, 5], 4) == [block_hashes([1, 2, 3, 4], 4)[0]]


def test_prefix_sharing_saves_recompute() -> None:
    c = TieredKVCache(block_size=4, bytes_per_block=64)
    seq = list(range(32))                       # 8 blocks
    c.insert(seq)
    out = c.insert(seq + [99, 99, 99, 99])      # one new sealed block
    assert out["blocks_cached"] == 8
    assert out["blocks_new"] == 1
    assert out["prefix_hit_tokens"] == 32


def test_disk_tier_persists_to_filesystem(tmp_path) -> None:
    c = TieredKVCache(
        block_size=4, bytes_per_block=100,
        gpu_bytes=200, cpu_bytes=200, disk_dir=tmp_path,
    )
    seq = list(range(80))                        # 20 blocks, 2000B >> 400B memory
    c.insert(seq)
    # Something must have spilled all the way to disk and been written out.
    assert c.stats.evicted_to_disk > 0
    assert any(tmp_path.glob("*.kv"))
    # Every block is still recoverable across the hierarchy.
    assert c.lookup_prefix(seq) == 80


def test_demotion_never_drops_blocks() -> None:
    c = TieredKVCache(block_size=4, bytes_per_block=100, gpu_bytes=300, cpu_bytes=300)
    seq = list(range(80))
    c.insert(seq)
    for t in (BlockTier.GPU, BlockTier.CPU):
        assert c.used_bytes(t) <= 300            # budgets respected
    assert c.lookup_prefix(seq) == 80            # nothing lost


def test_lookup_promotes_toward_gpu() -> None:
    c = TieredKVCache(block_size=4, bytes_per_block=100, gpu_bytes=300, cpu_bytes=300)
    seq = list(range(80))
    c.insert(seq)
    # The most-recently inserted blocks sit on GPU; older ones were demoted.
    c.lookup_prefix(seq[:8])                     # touch first 2 blocks
    assert c.tier_of(seq, 0) == BlockTier.GPU
    assert c.tier_of(seq, 1) == BlockTier.GPU


def test_oversized_block_rejected() -> None:
    c = TieredKVCache(block_size=4, gpu_bytes=50)
    try:
        c.insert(list(range(4)), payloads=[b"x" * 999])
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---- load balancer ---------------------------------------------------------

def test_lb_offline_invariants() -> None:
    ok, detail = load_balancer.offline_invariants()
    assert ok, detail


def test_affinity_keeps_prefix_on_same_worker() -> None:
    r = CacheAwareRouter(4, block_size=4)
    prompt = list(range(16))
    d0 = r.route(prompt)
    r.record_completion(d0.worker, prompt)
    d1 = r.route(prompt + [55, 56])
    assert d1.worker == d0.worker
    assert d1.reason == "affinity"
    assert d1.prefix_blocks_hit == 4


def test_cache_aware_improves_cluster_hit_rate() -> None:
    ok, detail = load_balancer.offline_invariants()
    assert detail["cache_aware_hit_rate"] > detail["round_robin_hit_rate"]

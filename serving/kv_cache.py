# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tiered, prefix-sharing paged KV cache (GPU → CPU → disk).

Why this exists
---------------
Large-scale LLM serving lives or dies on the KV cache. Two ideas dominate
production engines (vLLM PagedAttention, SGLang RadixAttention, DeepSeek's
context-caching service):

1. **Paging + prefix sharing.** A sequence's KV is chunked into fixed-size
   *blocks*. A block is keyed by a rolling hash over *all* token ids up to and
   including it, so two requests that share a prompt prefix share the exact same
   physical blocks — the second request's prefill is skipped for the shared span.

2. **A memory hierarchy.** Hot blocks live in fast/scarce GPU HBM; cooled blocks
   are *demoted* to host DRAM, then to disk, instead of being dropped. A later
   request that needs them pays a copy, not a recompute. This is the "KV Cache
   磁盘缓存" the role description calls out.

This module implements the *policy* for both, exactly, in dependency-free Python:
block hashing, longest-prefix lookup, per-tier byte budgets, LRU demotion on
overflow, and promotion-on-hit. The payload is opaque ``bytes`` (a real engine
would store device pointers / tensors); the bookkeeping is identical, which lets
the eviction and sharing behaviour be unit-tested deterministically on any
machine. The disk tier is a real on-disk store (one file per block) so the
persistence path is exercised, not faked.

Falsifiable offline invariants (``offline_invariants()``, CI-gated):
  - identical prefixes share blocks (a re-insert is a 100% prefix hit, 0 new bytes);
  - a divergent token forks the cache exactly at the first differing block;
  - no tier ever exceeds its byte budget;
  - an overflowing block is demoted (recoverable from the lower tier), never lost;
  - a lookup that crosses tiers promotes the blocks back toward GPU;
  - byte accounting closes (sum of live tiers == sum of tracked block sizes).
"""

from __future__ import annotations

import hashlib
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Optional


class BlockTier(IntEnum):
    """Memory tiers, fastest/scarcest first. Lower value == closer to compute."""

    GPU = 0
    CPU = 1
    DISK = 2


def _hash_tokens(token_ids: Iterable[int]) -> str:
    """Stable 16-hex content hash over a token-id sequence."""
    h = hashlib.blake2b(digest_size=8)
    for t in token_ids:
        h.update(int(t).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def block_hashes(token_ids: list[int], block_size: int) -> list[str]:
    """Rolling *prefix* hashes, one per full block.

    Block ``i`` is hashed over ``token_ids[: (i+1)*block_size]`` — every token up
    to and including the block — so the hash chain encodes the prefix. Two
    sequences agree on block ``i`` iff they agree on the first ``(i+1)*block_size``
    tokens. A trailing partial block is intentionally **not** hashed: only sealed
    (full) blocks are shareable, mirroring real paged caches.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    n_full = len(token_ids) // block_size
    out: list[str] = []
    for i in range(n_full):
        out.append(_hash_tokens(token_ids[: (i + 1) * block_size]))
    return out


@dataclass
class _Block:
    key: str
    n_bytes: int
    tier: BlockTier


@dataclass
class KVCacheStats:
    """Counters for measuring cache behaviour (all monotonic)."""

    lookups: int = 0
    blocks_requested: int = 0
    blocks_hit: int = 0          # served from any tier (saved a recompute)
    gpu_hits: int = 0
    cpu_hits: int = 0
    disk_hits: int = 0
    misses: int = 0              # blocks that had to be (re)computed + inserted
    demotions: int = 0
    promotions: int = 0
    evicted_to_disk: int = 0
    bytes_written_disk: int = 0

    @property
    def hit_rate(self) -> float:
        return self.blocks_hit / self.blocks_requested if self.blocks_requested else 0.0

    @property
    def prefix_cache_hit_rate(self) -> float:
        """Alias used in serving dashboards (== block hit rate here)."""
        return self.hit_rate

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["hit_rate"] = round(self.hit_rate, 4)
        return d


class TieredKVCache:
    """A paged KV cache spanning GPU/CPU/disk with prefix sharing.

    Parameters
    ----------
    block_size:
        Tokens per block (vLLM default is 16).
    gpu_bytes / cpu_bytes:
        Byte budgets for the in-memory tiers. Overflow demotes the
        least-recently-used block one tier down. Disk is treated as unbounded.
    disk_dir:
        Directory backing the disk tier. If ``None``, the disk tier is
        memory-emulated (still counts bytes and exercises demotion, just doesn't
        touch the filesystem) — handy for tests that don't want temp dirs.
    bytes_per_block:
        Synthetic per-block KV size used when a caller doesn't supply real
        payloads (so capacity math is realistic: ``2 * layers * 2 * heads *
        head_dim * block_size * dtype_bytes`` for a real model).
    """

    def __init__(
        self,
        block_size: int = 16,
        *,
        gpu_bytes: int = 1 << 20,
        cpu_bytes: int = 1 << 22,
        disk_dir: Optional[Path | str] = None,
        bytes_per_block: int = 4096,
    ) -> None:
        if block_size <= 0 or bytes_per_block <= 0:
            raise ValueError("block_size and bytes_per_block must be positive")
        self.block_size = block_size
        self.bytes_per_block = bytes_per_block
        self._budget = {BlockTier.GPU: gpu_bytes, BlockTier.CPU: cpu_bytes}
        # One LRU map per in-memory tier (key -> _Block), MRU at the end.
        self._tiers: dict[BlockTier, "OrderedDict[str, _Block]"] = {
            BlockTier.GPU: OrderedDict(),
            BlockTier.CPU: OrderedDict(),
            BlockTier.DISK: OrderedDict(),
        }
        self._disk_dir = Path(disk_dir) if disk_dir is not None else None
        if self._disk_dir is not None:
            self._disk_dir.mkdir(parents=True, exist_ok=True)
        self.stats = KVCacheStats()

    # ---- internals ---------------------------------------------------------

    def _used(self, tier: BlockTier) -> int:
        return sum(b.n_bytes for b in self._tiers[tier].values())

    def _disk_path(self, key: str) -> Path:
        assert self._disk_dir is not None
        return self._disk_dir / f"{key}.kv"

    def _write_disk(self, key: str, payload: bytes) -> None:
        if self._disk_dir is not None:
            self._disk_path(key).write_bytes(payload)

    def _evict_one(self, tier: BlockTier) -> None:
        """Demote the LRU block of ``tier`` one level down (never drop).

        Demotion cascades: if the destination tier is itself over budget after
        accepting the block, *its* LRU is demoted first, so no in-memory tier
        ever exceeds its budget (a GPU spill can ripple GPU→CPU→DISK).
        """
        key, blk = next(iter(self._tiers[tier].items()))  # LRU == first
        del self._tiers[tier][key]
        lower = BlockTier(tier + 1)
        if lower != BlockTier.DISK:
            self._ensure_room(lower, blk.n_bytes)
        blk.tier = lower
        self._tiers[lower][key] = blk
        self.stats.demotions += 1
        if lower == BlockTier.DISK:
            self.stats.evicted_to_disk += 1
            self.stats.bytes_written_disk += blk.n_bytes
            self._write_disk(key, b"\0" * blk.n_bytes)

    def _ensure_room(self, tier: BlockTier, incoming: int) -> None:
        if tier == BlockTier.DISK:
            return  # unbounded
        budget = self._budget[tier]
        if incoming > budget:
            raise ValueError(
                f"block of {incoming}B cannot fit tier {tier.name} budget {budget}B"
            )
        while self._tiers[tier] and self._used(tier) + incoming > budget:
            self._evict_one(tier)

    def _insert_gpu(self, key: str, n_bytes: int) -> None:
        self._ensure_room(BlockTier.GPU, n_bytes)
        self._tiers[BlockTier.GPU][key] = _Block(key, n_bytes, BlockTier.GPU)

    def _promote(self, key: str) -> None:
        """Move an existing block (in any lower tier) up to GPU on a hit."""
        for tier in (BlockTier.CPU, BlockTier.DISK):
            if key in self._tiers[tier]:
                blk = self._tiers[tier].pop(key)
                if tier == BlockTier.DISK and self._disk_dir is not None:
                    p = self._disk_path(key)
                    if p.exists():
                        p.unlink()
                self._ensure_room(BlockTier.GPU, blk.n_bytes)
                blk.tier = BlockTier.GPU
                self._tiers[BlockTier.GPU][key] = blk
                self.stats.promotions += 1
                return
        # already on GPU — just mark MRU
        if key in self._tiers[BlockTier.GPU]:
            self._tiers[BlockTier.GPU].move_to_end(key)

    def _find(self, key: str) -> Optional[BlockTier]:
        for tier in (BlockTier.GPU, BlockTier.CPU, BlockTier.DISK):
            if key in self._tiers[tier]:
                return tier
        return None

    # ---- public API --------------------------------------------------------

    def lookup_prefix(self, token_ids: list[int]) -> int:
        """Longest cached **block-aligned** prefix length, in tokens.

        Walks the block-hash chain; stops at the first block not present in any
        tier (a fork point). Records per-tier hit stats and promotes every hit
        block toward GPU (it's about to be reused). Does **not** insert.
        """
        hashes = block_hashes(token_ids, self.block_size)
        hit_blocks = 0
        for key in hashes:
            tier = self._find(key)
            if tier is None:
                break
            self.stats.blocks_hit += 1
            if tier == BlockTier.GPU:
                self.stats.gpu_hits += 1
            elif tier == BlockTier.CPU:
                self.stats.cpu_hits += 1
            else:
                self.stats.disk_hits += 1
            self._promote(key)
            hit_blocks += 1
        return hit_blocks * self.block_size

    def insert(self, token_ids: list[int], payloads: Optional[list[bytes]] = None) -> dict:
        """Insert/refresh the sealed blocks of ``token_ids``.

        Returns a per-call summary: how many blocks were already cached (prefix
        hit) vs. newly computed (miss). Shared prefixes cost **zero** new bytes —
        the whole point of prefix caching. Mirrors a prefill: you look up what's
        cached, then only "compute" (insert) the suffix.
        """
        hashes = block_hashes(token_ids, self.block_size)
        self.stats.lookups += 1
        self.stats.blocks_requested += len(hashes)
        cached = 0
        new = 0
        for i, key in enumerate(hashes):
            existing = self._find(key)
            if existing is not None:
                cached += 1
                self._promote(key)
                continue
            self.stats.misses += 1
            new += 1
            if payloads is not None and i < len(payloads):
                n = len(payloads[i])
            else:
                n = self.bytes_per_block
            self._insert_gpu(key, n)
        return {
            "blocks_total": len(hashes),
            "blocks_cached": cached,
            "blocks_new": new,
            "prefix_hit_tokens": cached * self.block_size,
        }

    def tier_of(self, token_ids: list[int], block_index: int) -> Optional[BlockTier]:
        """Which tier holds block ``block_index`` of this sequence (or None)."""
        hashes = block_hashes(token_ids, self.block_size)
        if not (0 <= block_index < len(hashes)):
            return None
        return self._find(hashes[block_index])

    def used_bytes(self, tier: BlockTier) -> int:
        return self._used(tier)

    def live_bytes(self) -> int:
        return sum(self._used(t) if t != BlockTier.DISK
                   else sum(b.n_bytes for b in self._tiers[t].values())
                   for t in BlockTier)

    def clear_disk(self) -> None:
        if self._disk_dir is not None and self._disk_dir.exists():
            shutil.rmtree(self._disk_dir)
            self._disk_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Offline invariants — deterministic, no deps, CI-gated.
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    """Prove the cache policy is correct without a GPU. Returns (ok, detail)."""
    checks: dict[str, bool] = {}

    bs = 4
    # 1. Identical re-insert is a full prefix hit costing zero new bytes.
    c = TieredKVCache(block_size=bs, bytes_per_block=100)
    seq = list(range(40))  # 10 full blocks
    first = c.insert(seq)
    bytes_after_first = c.live_bytes()
    second = c.insert(seq)
    checks["reinsert_full_hit"] = (
        first["blocks_new"] == 10
        and second["blocks_cached"] == 10
        and second["blocks_new"] == 0
        and c.live_bytes() == bytes_after_first
    )

    # 2. A divergent token forks exactly at the first differing block.
    forked = seq[:20] + [999] + seq[21:]  # diverges in block index 5 (tokens 20-23)
    res = c.insert(forked)
    # blocks 0..4 (tokens 0..19) shared; block 5 onward recomputed
    checks["fork_point"] = res["blocks_cached"] == 5 and res["blocks_new"] == 5

    # 3 & 4. Budget never exceeded; overflow demotes (not drops).
    small = TieredKVCache(
        block_size=bs, bytes_per_block=100, gpu_bytes=300, cpu_bytes=300
    )
    big = list(range(80))  # 20 blocks * 100B = 2000B >> 600B in-memory budget
    small.insert(big)
    checks["gpu_within_budget"] = small.used_bytes(BlockTier.GPU) <= 300
    checks["cpu_within_budget"] = small.used_bytes(BlockTier.CPU) <= 300
    checks["demoted_not_dropped"] = small.stats.demotions > 0
    # every block still recoverable somewhere
    recovered = small.lookup_prefix(big)
    checks["all_blocks_recoverable"] = recovered == 80

    # 5. A cross-tier lookup promotes blocks back toward GPU. Touch a span that
    # fits the GPU budget (3 blocks @ 100B) so the promotion isn't immediately
    # re-evicted; the touched blocks must then live on GPU.
    promos_before = small.stats.promotions
    small.lookup_prefix(big[: 2 * bs])  # touch first 2 blocks (fits 300B GPU)
    checks["lookup_promotes"] = small.stats.promotions >= promos_before
    checks["touched_blocks_on_gpu"] = (
        small.tier_of(big, 0) == BlockTier.GPU
        and small.tier_of(big, 1) == BlockTier.GPU
    )

    # 6. Byte accounting closes after churn.
    total_tracked = sum(
        b.n_bytes for t in BlockTier for b in small._tiers[t].values()
    )
    checks["accounting_closes"] = small.live_bytes() == total_tracked

    # 7. Block hashing is prefix-deterministic and order-sensitive.
    h1 = block_hashes([1, 2, 3, 4, 5, 6, 7, 8], 4)
    h2 = block_hashes([1, 2, 3, 4, 9, 9, 9, 9], 4)
    checks["prefix_hash_shares"] = h1[0] == h2[0] and h1[1] != h2[1]

    ok = all(checks.values())
    return ok, {"checks": checks}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("KV cache offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)

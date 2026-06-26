# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Tiered MoE expert offloading — only active experts resident in fast memory.

*Why this exists.* A large MoE (GLM-5.2: 744B total / 40B active) cannot fit its full
expert pool in GPU HBM, but at any instant only the ``k`` experts the router selected for
the current token actually need to be resident. The rest can sit in cheaper tiers (host
DRAM, then disk) and be *promoted* on demand when the router picks them. This is the
weight-space analog of the KV-cache tiering in :mod:`serving.kv_cache`: the same
GPU→CPU→disk demotion/promotion, applied to *experts* instead of *KV blocks*.

*Why it is governed.* This is the cleanest instance of the ``Governed-Scaling.md`` thesis:
an expert is *promoted* to fast memory **only when the router selects it** — the
"promote-only-what-verifies" governor applied to *infrastructure state*. The router's
selection is the "verification" that the expert is needed; nothing is resident without that
signal. The tier transitions mirror ``TieredKVCache`` exactly (LRU demotion on overflow,
promotion-on-hit), so the governance and the bookkeeping are the same object.

*What it is not.* A reference implementation of the *policy* (which tier, when to demote,
promote-on-route), dependency-free Python, CI-tested — exactly like :mod:`serving.kv_cache`.
It does not move real GPU tensors; the expert payload is opaque bytes. The deployment
artifact is a fused expert-load kernel + the routing glue.

Honest scope (see ``docs/11-Platform/Cheap-Compute-Boundary.md`` Boundary 3): this delivers
"tiered expert memory with promote-on-route governance." A "low-RAM at release" *capability*
claim still needs the quantized experts evaluated against FP16 on a held-out set to the
no-overclaim gate — this module is the *mechanism*, not the *measurement*.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class ExpertTier(IntEnum):
    """Memory tiers for expert weights, fastest/scarcest first (mirrors BlockTier)."""

    GPU = 0   # resident, ready to compute
    CPU = 1   # host DRAM, one copy away
    DISK = 2  # cold storage, slowest


@dataclass
class _ExpertEntry:
    expert_id: int
    size_bytes: int
    tier: ExpertTier
    # LRU recency, shared across tiers so the global eviction order is meaningful.
    last_used: int = 0


@dataclass
class ExpertOffloadStats:
    """Bookkeeping for the governed-scaling meter — measurable, not asserted."""

    promotes: int = 0       # expert moved UP a tier (a route demanded it)
    demotes: int = 0        # expert moved DOWN (GPU overflowed)
    gpu_hits: int = 0       # route found the expert already on GPU (no copy)
    disk_loads: int = 0     # expert had to come all the way from disk
    gpu_resident_bytes: int = 0
    gpu_budget_bytes: int = 0

    @property
    def gpu_utilization(self) -> float:
        return (self.gpu_resident_bytes / self.gpu_budget_bytes) if self.gpu_budget_bytes else 0.0


class TieredExpertStore:
    """A 3-tier expert weight store with promote-on-route governance.

    Experts start wherever you place them (default DISK). When :meth:`route_select` is
    called with the router's chosen expert ids, any selected expert not on GPU is
    *promoted* there (demoting LRU GPU experts to CPU→disk if the GPU budget overflows),
    so only the *active* expert set is ever resident in fast memory. This is the
    load-bearing low-RAM mechanism: a 744B MoE keeps ~40B active worth of experts on GPU.

    Tiers are bounded LRU ``OrderedDict`` (expert_id -> _ExpertEntry); the GPU tier has a
    byte budget, CPU and disk are capped by count for determinism in tests.
    """

    def __init__(self, *, gpu_budget_bytes: int, cpu_capacity: int = 64,
                 disk_capacity: int = 4096) -> None:
        if gpu_budget_bytes <= 0:
            raise ValueError("gpu_budget_bytes must be positive")
        self.gpu_budget_bytes = gpu_budget_bytes
        self.cpu_capacity = cpu_capacity
        self.disk_capacity = disk_capacity
        # expert_id -> _ExpertEntry (single source of truth for tier + size)
        self._entries: dict[int, _ExpertEntry] = {}
        # per-tier LRU orderings (OrderedDict for move-to-end on access)
        self._tiers: dict[ExpertTier, OrderedDict[int, None]] = {
            ExpertTier.GPU: OrderedDict(),
            ExpertTier.CPU: OrderedDict(),
            ExpertTier.DISK: OrderedDict(),
        }
        self._tick = 0
        self.stats = ExpertOffloadStats(gpu_budget_bytes=gpu_budget_bytes)

    # -- registration ---------------------------------------------------------

    def register(self, expert_id: int, size_bytes: int, *,
                 tier: ExpertTier = ExpertTier.DISK) -> None:
        """Add an expert to the store at a starting tier (default cold/disk)."""
        if expert_id in self._entries:
            raise ValueError(f"expert {expert_id} already registered")
        if size_bytes <= 0:
            raise ValueError("size_bytes must be positive")
        self._entries[expert_id] = _ExpertEntry(expert_id, size_bytes, tier)
        self._tiers[tier][expert_id] = None
        if tier == ExpertTier.GPU:
            self.stats.gpu_resident_bytes += size_bytes
        self._enforce_gpu_budget()

    # -- the governed primitive: promote-on-route -----------------------------

    def route_select(self, expert_ids: "list[int]") -> "dict[str, object]":
        """The router selected these experts. Ensure they are GPU-resident.

        This is the promote-on-route governor: every selected expert is promoted to GPU
        (demoting LRU victims to make room), so the *active* set is always ready and the
        inactive set costs no fast memory. Returns a report of what moved.

        ``expert_ids`` should be the unique set of experts the router picked this step
        (e.g. the union over the token batch's top-k assignments).
        """
        self._tick += 1
        moved_up = 0
        for eid in expert_ids:
            if eid not in self._entries:
                raise KeyError(f"expert {eid} not registered")
            entry = self._entries[eid]
            entry.last_used = self._tick
            if entry.tier == ExpertTier.GPU:
                self.stats.gpu_hits += 1
                self._tiers[ExpertTier.GPU].move_to_end(eid)
                continue
            # promote: DISK→CPU→GPU (or CPU→GPU), counting each crossing
            self._promote_to_gpu(eid)
            moved_up += 1
        return {
            "requested": list(expert_ids),
            "moved_up": moved_up,
            "gpu_resident_bytes": self.stats.gpu_resident_bytes,
            "gpu_utilization": round(self.stats.gpu_utilization, 4),
        }

    def _promote_to_gpu(self, expert_id: int) -> None:
        """Move ``expert_id`` to GPU, demoting LRU victims to make budget room."""
        entry = self._entries[expert_id]
        prev_tier = entry.tier
        # count the tier crossings (DISK→CPU→GPU = 2 promotes; CPU→GPU = 1)
        if prev_tier == ExpertTier.DISK:
            self._move(expert_id, ExpertTier.DISK, ExpertTier.CPU)
            self.stats.disk_loads += 1
            self.stats.promotes += 1
            prev_tier = ExpertTier.CPU
        if prev_tier == ExpertTier.CPU:
            self._move(expert_id, ExpertTier.CPU, ExpertTier.GPU)
            self.stats.promotes += 1
        self._enforce_gpu_budget()

    def _move(self, expert_id: int, frm: ExpertTier, to: ExpertTier) -> None:
        """Move an expert between adjacent tiers, updating LRU and byte accounting."""
        entry = self._entries[expert_id]
        assert entry.tier == frm, f"expected {frm}, expert on {entry.tier}"
        del self._tiers[frm][expert_id]
        entry.tier = to
        self._tiers[to][expert_id] = None
        if to == ExpertTier.GPU and frm != ExpertTier.GPU:
            self.stats.gpu_resident_bytes += entry.size_bytes
        if frm == ExpertTier.GPU and to != ExpertTier.GPU:
            self.stats.gpu_resident_bytes -= entry.size_bytes

    def _enforce_gpu_budget(self) -> None:
        """Demote LRU GPU experts to CPU (and CPU→disk on overflow) until under budget."""
        while self.stats.gpu_resident_bytes > self.gpu_budget_bytes and self._tiers[ExpertTier.GPU]:
            # Peek the LRU key; _move owns the dict removal (avoids double-delete).
            victim = next(iter(self._tiers[ExpertTier.GPU]))
            self._move(victim, ExpertTier.GPU, ExpertTier.CPU)
            self.stats.demotes += 1
            # cascade CPU→disk if CPU over capacity
            while len(self._tiers[ExpertTier.CPU]) > self.cpu_capacity and self._tiers[ExpertTier.CPU]:
                v2 = next(iter(self._tiers[ExpertTier.CPU]))
                self._move(v2, ExpertTier.CPU, ExpertTier.DISK)
                if len(self._tiers[ExpertTier.DISK]) > self.disk_capacity:
                    # disk overflow: drop the coldest (would be re-fetched on demand)
                    v3 = next(iter(self._tiers[ExpertTier.DISK]))
                    del self._tiers[ExpertTier.DISK][v3]
                    del self._entries[v3]

    # -- introspection --------------------------------------------------------

    def tier_of(self, expert_id: int) -> ExpertTier:
        return self._entries[expert_id].tier

    def gpu_resident(self) -> "list[int]":
        return list(self._tiers[ExpertTier.GPU].keys())


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. Only routed experts end up on GPU; the rest stay cold (the low-RAM guarantee).
    store = TieredExpertStore(gpu_budget_bytes=300)   # room for ~3 small experts
    for eid in range(8):
        store.register(eid, size_bytes=100, tier=ExpertTier.DISK)
    store.route_select([0, 1])                         # router picks experts 0,1
    checks["routed_resident"] = set(store.gpu_resident()) == {0, 1}
    checks["rest_cold"] = all(store.tier_of(e) == ExpertTier.DISK for e in range(2, 8))
    detail["after_first_route"] = store.gpu_resident()

    # 2. GPU never exceeds its byte budget (LRU demotion enforces it).
    store.route_select([2, 3, 4])                      # would need 500 bytes > 300 budget
    checks["budget_respected"] = store.stats.gpu_resident_bytes <= 300
    checks["at_most_three_resident"] = len(store.gpu_resident()) <= 3
    detail["after_overflow_route"] = store.gpu_resident()
    detail["demotes"] = store.stats.demotes

    # 3. Re-routing an already-resident expert is a GPU hit (no promote, no copy).
    hits_before = store.stats.gpu_hits
    resident = store.gpu_resident()
    store.route_select(resident[:1])
    checks["reroute_is_gpu_hit"] = store.stats.gpu_hits == hits_before + 1

    # 4. Promotion crosses tiers in order (DISK→CPU→GPU), counting each step.
    s2 = TieredExpertStore(gpu_budget_bytes=1000)
    s2.register(0, size_bytes=100, tier=ExpertTier.DISK)
    rep = s2.route_select([0])
    checks["promoted_from_disk"] = s2.tier_of(0) == ExpertTier.GPU
    # DISK→CPU→GPU is 2 promotes + 1 disk_load
    checks["disk_load_counted"] = s2.stats.disk_loads == 1
    detail["promotes_for_disk_expert"] = s2.stats.promotes

    # 5. A 744B/40B-style ratio: 8 experts, GPU fits 2 (active set), routing thrashes.
    s3 = TieredExpertStore(gpu_budget_bytes=200)       # 2 of 8 experts resident
    for e in range(8):
        s3.register(e, size_bytes=100, tier=ExpertTier.DISK)
    # simulate 5 decode steps, router picks 2 different experts each time
    for step in range(5):
        s3.route_select([(step * 2) % 8, (step * 2 + 1) % 8])
    checks["thrash_stays_in_budget"] = s3.stats.gpu_resident_bytes <= 200
    checks["thrash_promotes_happen"] = s3.stats.promotes >= 5   # real offload churn
    detail["thrash_stats"] = {
        "promotes": s3.stats.promotes, "demotes": s3.stats.demotes,
        "gpu_hits": s3.stats.gpu_hits, "disk_loads": s3.stats.disk_loads,
    }

    # 6. Duplicate registration and unknown expert are rejected (fail-closed).
    s4 = TieredExpertStore(gpu_budget_bytes=100)
    s4.register(0, size_bytes=10)
    try:
        s4.register(0, size_bytes=10); checks["dup_rejected"] = False
    except ValueError:
        checks["dup_rejected"] = True
    try:
        s4.route_select([99]); checks["unknown_rejected"] = False
    except KeyError:
        checks["unknown_rejected"] = True

    # 7. Zero/negative budget rejected.
    try:
        TieredExpertStore(gpu_budget_bytes=0); checks["bad_budget_rejected"] = False
    except ValueError:
        checks["bad_budget_rejected"] = True

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Expert-offload offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  thrash: {detail.get('thrash_stats')}")
    raise SystemExit(0 if ok else 1)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia serving systems — the inference-infrastructure track.

This package is the *systems-engineering* counterpart to the trust layer: it
demonstrates the large-scale-inference primitives a training/inference-framework
role cares about, built to the same measurement discipline as the rest of the
repo (deterministic offline invariants in CI; expensive/GPU paths gated and
clearly labelled, never silently asserted).

Modules
-------
- ``kv_cache``      : tiered (GPU→CPU→disk) paged KV cache with prefix sharing,
                      block-hash dedup, LRU demotion/promotion, byte accounting.
- ``load_balancer`` : cache-aware request router — longest-prefix affinity over a
                      worker fleet, bounded by a load-imbalance cap, with a
                      consistent-hash fallback for cold prefixes.
- ``expert_offload`` : tiered (GPU→CPU→disk) MoE expert weight store with
                      promote-on-route governance — only the active expert set is
                      resident in fast memory (the weight-space analog of the KV
                      tiering). The low-RAM mechanism for serving a large MoE.
- ``kv_quant``       : INT8/INT4 KV-cache quantization with a content-deterministic
                      per-block scale, so prefix sharing survives quantization;
                      round-trip error bounded.

Honest scope: these are reference implementations in pure Python. They model the
*policy* (what to cache, where to evict, how to route, when to promote/offload)
exactly; they do not move real GPU tensors. The payloads are opaque bytes, so the
eviction/affinity/offload logic is identical to a production engine while staying
CI-testable on any machine. See ``docs/SYSTEMS-TRACK.md`` and
``docs/11-Platform/Cheap-Compute-Boundary.md``.
"""

from __future__ import annotations

from serving.expert_offload import (
    ExpertOffloadStats,
    ExpertTier,
    TieredExpertStore,
)
from serving.kv_cache import (
    BlockTier,
    KVCacheStats,
    TieredKVCache,
    block_hashes,
)
from serving.kv_quant import (
    dequantize_kv_block,
    kv_memory_ratio,
    quantize_kv_block,
)
from serving.load_balancer import CacheAwareRouter, RouteDecision

__all__ = [
    "BlockTier",
    "KVCacheStats",
    "TieredKVCache",
    "block_hashes",
    "CacheAwareRouter",
    "RouteDecision",
    # expert offloading (governed)
    "ExpertTier",
    "ExpertOffloadStats",
    "TieredExpertStore",
    # KV cache quantization
    "quantize_kv_block",
    "dequantize_kv_block",
    "kv_memory_ratio",
]

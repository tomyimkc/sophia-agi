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
- ``layer_stream``   : layer-by-layer (GPU→CPU→disk) dense-weight streaming with a
                      prefetch window and quant-aware sizing — the dense analog of
                      ``expert_offload`` and the AirLLM technique (run a model larger
                      than fast memory; ``tools/shard_checkpoint.py`` produces the
                      on-disk shards it streams).
- ``lowram_eval``    : the no-overclaim gate for a low-RAM deployment — measures the
                      streamed+quantized model against FP16 on a held-out set
                      (bounded KL / top-1 agreement, protected floor) so byte savings
                      never silently cost unmeasured quality.
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
from serving.layer_stream import (
    LayerStreamStats,
    LayerTier,
    StreamingLayerStore,
    plan_layer_bits,
    resident_bytes_for,
)
from serving.lowram_eval import (
    LowRamGate,
    LowRamReport,
)
from serving.kv_quant import (
    dequantize_kv_block,
    kv_memory_ratio,
    quantize_kv_block,
)
from serving.load_balancer import CacheAwareRouter, RouteDecision
from serving.gss_feasibility import (
    GSSFeasibilityGate,
    GSSFeasibilityReport,
    acceptance_rate,
    aggregate_runs,
    bootstrap_ci,
    expected_accepted,
    feasibility_with_ci,
    read_set_fraction,
    read_set_temporal_stability,
)
from serving.gss import (
    GSSEquivalenceGate,
    GSSEquivalenceReport,
    read_set_mask,
    speculative_realized,
    verify_drift,
)

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
    # layer-by-layer weight streaming (AirLLM-style)
    "LayerTier",
    "LayerStreamStats",
    "StreamingLayerStore",
    "plan_layer_bits",
    "resident_bytes_for",
    # low-RAM measurement gate
    "LowRamGate",
    "LowRamReport",
    # KV cache quantization
    "quantize_kv_block",
    "dequantize_kv_block",
    "kv_memory_ratio",
    # Governed Speculative Sparsity — Tier-0 feasibility meter + CIs
    "GSSFeasibilityGate",
    "GSSFeasibilityReport",
    "read_set_fraction",
    "read_set_temporal_stability",
    "acceptance_rate",
    "expected_accepted",
    "bootstrap_ci",
    "feasibility_with_ci",
    "aggregate_runs",
    # Governed Speculative Sparsity — Tier-1 mechanism + equivalence gate
    "speculative_realized",
    "verify_drift",
    "read_set_mask",
    "GSSEquivalenceGate",
    "GSSEquivalenceReport",
]

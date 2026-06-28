# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Layer-by-layer weight streaming — run a model larger than fast memory (分层流式推理).

*Why this exists.* A dense transformer is a stack of ``N`` near-identical layers executed
in sequence. At any instant only *one* layer's weights need to be resident to compute its
output — exactly the observation AirLLM (lyogavin/airllm) uses to run a 70B model on a 4GB
GPU. The model is decomposed into per-layer shards on disk; the forward pass streams one
layer at a time: load layer ``i`` → compute → free → load layer ``i+1``. Peak weight memory
is ``max(single layer) + activations + KV``, not the whole model. The cost is wall-clock and
disk I/O (every token re-streams every layer), so this is the *serving* low-RAM lever, not a
change to the trained artifact.

*Relationship to the rest of the repo.* This is the **dense-weight analog** of
:mod:`serving.expert_offload`: the same ``GPU→CPU→disk`` tiering, LRU demotion and
byte accounting, but the promotion signal is **sequential layer order** (or a prefetch
window) instead of an MoE router's top-k selection. Where ``expert_offload`` keeps the
*active expert set* resident, this keeps the *current layer window* resident. The two
compose: an MoE's dense trunk streams here, its experts tier in ``expert_offload``.

*What it adds over a naive stream.*
  1. **Prefetch (double-buffer).** While computing layer ``i``, layers ``i+1..i+depth`` are
     promoted ahead of time, so their ``step`` is a hit, not a stall — AirLLM's "overlap the
     model loading and compute" (v2.5+). Measured as ``prefetch_hits``.
  2. **Quant-aware sizing.** A layer registered at ``bits`` < 16 occupies
     ``ceil(fp16_bytes · bits/16)`` resident bytes, so the same GPU budget holds more layers
     (or a bigger model fits). Pair with :func:`plan_layer_bits` (delegates to
     :func:`moe.adapt.bit_allocator`) to spend bits where they matter.

*What it is not.* A reference implementation of the **policy** (tier transitions, prefetch
window, eviction, byte/quant accounting), dependency-free Python, CI-tested — exactly like
:mod:`serving.expert_offload` and :mod:`serving.kv_cache`. It does not move real GPU tensors;
the layer payload is opaque (a size in bytes). The deployment artifact is the mmap/safetensors
loader + a CUDA-stream prefetch kernel; the shard layout it consumes is produced by
``tools/shard_checkpoint.py``.

Honest scope (mirrors ``serving/expert_offload.py``): this is the *mechanism*. A "runs a 70B
model in <4GB" capability claim still needs the streamed+quantized model evaluated against
FP16 on a held-out set to the no-overclaim gate — that measurement lives in
:mod:`serving.lowram_eval`, not here.

Falsifiable offline invariants (``offline_invariants()``, CI-gated):
  - a full sequential pass keeps peak GPU bytes within budget (the low-RAM guarantee);
  - with a window wide enough for the prefetch depth, each layer is loaded from disk
    **exactly once per pass** (no thrash — AirLLM's stream-once property);
  - prefetch turns the next layers' steps into hits (overlap is real, not asserted);
  - quant-aware sizing lets a fixed budget hold strictly more layers resident.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional


class LayerTier(IntEnum):
    """Memory tiers for layer weights, fastest/scarcest first (mirrors ExpertTier)."""

    GPU = 0   # resident, ready to compute
    CPU = 1   # host DRAM, one copy away
    DISK = 2  # cold storage, slowest


@dataclass
class _LayerEntry:
    layer_idx: int
    resident_bytes: int       # size at its quantized width (what GPU residence costs)
    bits: int                 # quantized width (16 == fp16 baseline)
    tier: LayerTier
    last_used: int = 0        # LRU recency, shared across tiers


@dataclass
class LayerStreamStats:
    """Bookkeeping for the streaming meter — measurable, not asserted."""

    promotes: int = 0          # layer moved UP a tier
    demotes: int = 0           # layer moved DOWN (GPU overflowed)
    gpu_hits: int = 0          # step found the layer already on GPU (no copy)
    prefetch_hits: int = 0     # the hit was thanks to a prior prefetch (overlap paid off)
    disk_loads: int = 0        # layer had to come all the way from disk
    prefetched: int = 0        # layers promoted ahead of their own step
    gpu_resident_bytes: int = 0
    gpu_budget_bytes: int = 0
    peak_gpu_bytes: int = 0    # high-water mark over the run (the number you size HBM to)

    @property
    def gpu_utilization(self) -> float:
        return (self.gpu_resident_bytes / self.gpu_budget_bytes) if self.gpu_budget_bytes else 0.0


def resident_bytes_for(fp16_bytes: int, bits: int) -> int:
    """Resident cost of a layer stored at ``bits`` width, relative to its fp16 size.

    ``bits == 16`` is the identity (fp16 baseline). NVFP4 at ~4.5 effective bits is
    ~3.56× smaller (cf. :func:`moe.quant.nvfp4_memory_reduction`); we take the *nominal*
    ``bits`` here and let the caller pass an effective width if they want the micro-scale
    overhead counted. Rounds up so a partial byte never under-counts residence.
    """
    if fp16_bytes <= 0:
        raise ValueError("fp16_bytes must be positive")
    if not (1 <= bits <= 16):
        raise ValueError("bits must be in [1, 16]")
    return max(1, math.ceil(fp16_bytes * bits / 16))


class StreamingLayerStore:
    """A 3-tier per-layer weight store that streams a model larger than GPU memory.

    Layers start cold (DISK by default). :meth:`step` makes a layer GPU-resident for its
    compute and *prefetches* the next ``prefetch_depth`` layers, demoting the LRU tail to
    stay within ``gpu_budget_bytes``. A full forward pass is :meth:`forward_pass`, which
    walks ``0..num_layers-1`` and yields each layer the instant it is resident.

    Tiers are bounded LRU ``OrderedDict``; the GPU tier has a byte budget, CPU/disk are
    capped by count for deterministic tests (mirrors :class:`TieredExpertStore`).
    """

    def __init__(self, *, gpu_budget_bytes: int, prefetch_depth: int = 1,
                 cpu_capacity: int = 256, disk_capacity: int = 8192) -> None:
        if gpu_budget_bytes <= 0:
            raise ValueError("gpu_budget_bytes must be positive")
        if prefetch_depth < 0:
            raise ValueError("prefetch_depth must be >= 0")
        self.gpu_budget_bytes = gpu_budget_bytes
        self.prefetch_depth = prefetch_depth
        self.cpu_capacity = cpu_capacity
        self.disk_capacity = disk_capacity
        self._entries: dict[int, _LayerEntry] = {}
        self._tiers: dict[LayerTier, "OrderedDict[int, None]"] = {
            LayerTier.GPU: OrderedDict(),
            LayerTier.CPU: OrderedDict(),
            LayerTier.DISK: OrderedDict(),
        }
        # layers promoted by prefetch but not yet stepped — so the step can credit overlap.
        self._prefetched_pending: set[int] = set()
        self._tick = 0
        self.stats = LayerStreamStats(gpu_budget_bytes=gpu_budget_bytes)

    # -- registration ---------------------------------------------------------

    def register(self, layer_idx: int, fp16_bytes: int, *, bits: int = 16,
                 tier: LayerTier = LayerTier.DISK) -> None:
        """Add a layer to the store at a starting tier (default cold/disk).

        ``fp16_bytes`` is the layer's fp16 size; ``bits`` its quantized width. Residence
        costs ``resident_bytes_for(fp16_bytes, bits)`` — the quant-aware sizing that lets a
        fixed GPU budget hold more layers.
        """
        if layer_idx in self._entries:
            raise ValueError(f"layer {layer_idx} already registered")
        rbytes = resident_bytes_for(fp16_bytes, bits)
        self._entries[layer_idx] = _LayerEntry(layer_idx, rbytes, bits, tier)
        self._tiers[tier][layer_idx] = None
        if tier == LayerTier.GPU:
            self.stats.gpu_resident_bytes += rbytes
        self._enforce_gpu_budget()

    # -- the streaming primitive: step + prefetch -----------------------------

    def step(self, layer_idx: int) -> "dict[str, object]":
        """Compute on ``layer_idx``: make it GPU-resident, then prefetch the window ahead.

        Returns a report of what moved. The promotion signal here is sequential order —
        the analog of ``expert_offload.route_select`` for dense layers.
        """
        if layer_idx not in self._entries:
            raise KeyError(f"layer {layer_idx} not registered")
        self._tick += 1
        entry = self._entries[layer_idx]
        entry.last_used = self._tick
        if entry.tier == LayerTier.GPU:
            self.stats.gpu_hits += 1
            if layer_idx in self._prefetched_pending:
                self.stats.prefetch_hits += 1   # the hit was bought by an earlier prefetch
            self._tiers[LayerTier.GPU].move_to_end(layer_idx)
        else:
            self._promote_to_gpu(layer_idx)
        self._prefetched_pending.discard(layer_idx)

        # Prefetch the next `prefetch_depth` registered layers (overlap load with compute).
        prefetched = []
        nxt = layer_idx + 1
        while len(prefetched) < self.prefetch_depth and nxt in self._entries:
            pe = self._entries[nxt]
            if pe.tier != LayerTier.GPU:
                pe.last_used = self._tick   # newer than the just-computed layer (stays resident)
                self._promote_to_gpu(nxt)
                self.stats.prefetched += 1
                self._prefetched_pending.add(nxt)
                prefetched.append(nxt)
            nxt += 1

        self.stats.peak_gpu_bytes = max(self.stats.peak_gpu_bytes, self.stats.gpu_resident_bytes)
        return {
            "layer": layer_idx,
            "prefetched": prefetched,
            "gpu_resident_bytes": self.stats.gpu_resident_bytes,
            "gpu_utilization": round(self.stats.gpu_utilization, 4),
        }

    def forward_pass(self, num_layers: int):
        """Generator: walk ``0..num_layers-1`` in order, yielding each layer once resident.

        This is one full forward pass — the unit AirLLM pays per token. ``num_layers`` must
        not exceed the registered layer count.
        """
        if num_layers <= 0 or num_layers > len(self._entries):
            raise ValueError("num_layers must be in [1, #registered]")
        for i in range(num_layers):
            self.step(i)
            yield i

    def _promote_to_gpu(self, layer_idx: int) -> None:
        """Move ``layer_idx`` to GPU, demoting LRU victims to make budget room."""
        entry = self._entries[layer_idx]
        prev_tier = entry.tier
        if prev_tier == LayerTier.DISK:
            self._move(layer_idx, LayerTier.DISK, LayerTier.CPU)
            self.stats.disk_loads += 1
            self.stats.promotes += 1
            prev_tier = LayerTier.CPU
        if prev_tier == LayerTier.CPU:
            self._move(layer_idx, LayerTier.CPU, LayerTier.GPU)
            self.stats.promotes += 1
        self._enforce_gpu_budget()

    def _move(self, layer_idx: int, frm: LayerTier, to: LayerTier) -> None:
        """Move a layer between adjacent tiers, updating LRU and byte accounting."""
        entry = self._entries[layer_idx]
        assert entry.tier == frm, f"expected {frm}, layer on {entry.tier}"
        del self._tiers[frm][layer_idx]
        entry.tier = to
        self._tiers[to][layer_idx] = None
        if to == LayerTier.GPU and frm != LayerTier.GPU:
            self.stats.gpu_resident_bytes += entry.resident_bytes
        if frm == LayerTier.GPU and to != LayerTier.GPU:
            self.stats.gpu_resident_bytes -= entry.resident_bytes

    def _enforce_gpu_budget(self) -> None:
        """Demote LRU GPU layers to CPU (and CPU→disk on overflow) until under budget."""
        while self.stats.gpu_resident_bytes > self.gpu_budget_bytes and self._tiers[LayerTier.GPU]:
            victim = next(iter(self._tiers[LayerTier.GPU]))
            self._prefetched_pending.discard(victim)
            self._move(victim, LayerTier.GPU, LayerTier.CPU)
            self.stats.demotes += 1
            while len(self._tiers[LayerTier.CPU]) > self.cpu_capacity and self._tiers[LayerTier.CPU]:
                v2 = next(iter(self._tiers[LayerTier.CPU]))
                self._move(v2, LayerTier.CPU, LayerTier.DISK)
                if len(self._tiers[LayerTier.DISK]) > self.disk_capacity:
                    v3 = next(iter(self._tiers[LayerTier.DISK]))
                    del self._tiers[LayerTier.DISK][v3]
                    del self._entries[v3]

    # -- introspection --------------------------------------------------------

    def tier_of(self, layer_idx: int) -> LayerTier:
        return self._entries[layer_idx].tier

    def gpu_resident(self) -> "list[int]":
        return list(self._tiers[LayerTier.GPU].keys())


# ---------------------------------------------------------------------------
# Quant-aware bit planning — delegate to the repo's sensitivity allocator
# ---------------------------------------------------------------------------

def plan_layer_bits(layer_fp16_bytes: "dict[int, int]", target_avg_bits: float,
                    *, protected: "Optional[set[int]]" = None,
                    sensitivities: "Optional[dict[int, float]]" = None) -> "dict[int, int]":
    """Per-layer bit-widths under a target average width, via :func:`moe.adapt.bit_allocator`.

    The greedy sensitivity allocator already in ``moe/adapt.py`` is the right tool: spend
    bits where quantization hurts most, starve redundant layers, keep a protected floor on
    the layers that must stay high-precision. Here a "tensor" is a whole layer.

    ``layer_fp16_bytes`` : layer index -> fp16 byte size (the numel proxy / cost weight).
    ``target_avg_bits``  : byte-weighted average width to hit (e.g. 4.5 for the NVFP4 path).
    ``protected``        : layer indices never taken below the floor (e.g. embedding/head
                           layers if you fold them in; first/last blocks).
    ``sensitivities``    : optional measured per-layer output-KL sensitivity; absent, every
                           layer is weighted equally and the allocator distributes uniformly
                           under the byte budget.

    Returns layer index -> integer bit-width. Requires ``moe.adapt`` (numpy).
    """
    from moe.adapt import bit_allocator  # local import: numpy only needed on this path

    protected = protected or set()
    profiles = []
    for idx, fp16 in sorted(layer_fp16_bytes.items()):
        numels = max(1, fp16 // 2)                      # fp16 bytes -> element count proxy
        sens = (sensitivities or {}).get(idx, 1.0)
        profiles.append((str(idx), numels, float(sens), idx in protected))
    alloc = bit_allocator(profiles, target_avg_bits)
    return {int(name): bits for name, bits in alloc.items()}


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. A full sequential pass keeps peak GPU bytes within budget (the low-RAM guarantee).
    #    32 layers of 100 bytes = 3200 fp16; GPU budget of 300 holds ~3 — yet the pass runs.
    store = StreamingLayerStore(gpu_budget_bytes=300, prefetch_depth=1)
    for i in range(32):
        store.register(i, fp16_bytes=100, tier=LayerTier.DISK)
    for _ in store.forward_pass(32):
        pass
    checks["pass_within_budget"] = store.stats.peak_gpu_bytes <= 300
    checks["peak_far_below_model"] = store.stats.peak_gpu_bytes < 32 * 100  # << whole model
    detail["peak_gpu_bytes"] = store.stats.peak_gpu_bytes
    detail["whole_model_bytes"] = 32 * 100

    # 2. Stream-once: with a window wide enough for the prefetch depth, each layer loads
    #    from disk exactly once across a full pass (no thrash — AirLLM's property).
    wide = StreamingLayerStore(gpu_budget_bytes=400, prefetch_depth=1)  # holds >= depth+2
    for i in range(16):
        wide.register(i, fp16_bytes=100, tier=LayerTier.DISK)
    for _ in wide.forward_pass(16):
        pass
    checks["stream_once_per_pass"] = wide.stats.disk_loads == 16
    detail["disk_loads_for_16_layers"] = wide.stats.disk_loads

    # 3. Prefetch overlap is real: the layers ahead are already resident when stepped.
    checks["prefetch_overlap_paid"] = wide.stats.prefetch_hits >= 15  # all but layer 0
    detail["prefetch_hits"] = wide.stats.prefetch_hits

    # 4. Quant-aware sizing lets a fixed budget hold strictly more layers resident.
    fp16_store = StreamingLayerStore(gpu_budget_bytes=320, prefetch_depth=3)
    q_store = StreamingLayerStore(gpu_budget_bytes=320, prefetch_depth=3)
    for i in range(16):
        fp16_store.register(i, fp16_bytes=100, bits=16)            # 100 bytes resident each
        q_store.register(i, fp16_bytes=100, bits=4)               # ~25 bytes resident each
    fp16_store.step(0)
    q_store.step(0)
    checks["quant_holds_more_layers"] = len(q_store.gpu_resident()) > len(fp16_store.gpu_resident())
    detail["fp16_resident"] = len(fp16_store.gpu_resident())
    detail["int4_resident"] = len(q_store.gpu_resident())

    # 5. resident_bytes_for is the bit ratio, rounded up, identity at fp16.
    checks["sizing_identity_fp16"] = resident_bytes_for(100, 16) == 100
    checks["sizing_int4_quarter"] = resident_bytes_for(100, 4) == 25
    checks["sizing_rounds_up"] = resident_bytes_for(10, 3) == math.ceil(10 * 3 / 16)

    # 6. Re-stepping a resident layer is a GPU hit (no promote, no copy).
    s = StreamingLayerStore(gpu_budget_bytes=1000, prefetch_depth=0)
    s.register(0, fp16_bytes=100)
    s.step(0)
    hits = s.stats.gpu_hits
    s.step(0)
    checks["restep_is_gpu_hit"] = s.stats.gpu_hits == hits + 1

    # 7. Fail-closed: bad budget, bad bits, dup register, unknown layer.
    try:
        StreamingLayerStore(gpu_budget_bytes=0); checks["bad_budget_rejected"] = False
    except ValueError:
        checks["bad_budget_rejected"] = True
    try:
        resident_bytes_for(100, 0); checks["bad_bits_rejected"] = False
    except ValueError:
        checks["bad_bits_rejected"] = True
    s2 = StreamingLayerStore(gpu_budget_bytes=100)
    s2.register(0, fp16_bytes=10)
    try:
        s2.register(0, fp16_bytes=10); checks["dup_rejected"] = False
    except ValueError:
        checks["dup_rejected"] = True
    try:
        s2.step(99); checks["unknown_rejected"] = False
    except KeyError:
        checks["unknown_rejected"] = True

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Layer-stream offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  peak GPU bytes {detail.get('peak_gpu_bytes')} vs whole-model "
          f"{detail.get('whole_model_bytes')} (16-layer disk loads: "
          f"{detail.get('disk_loads_for_16_layers')}, prefetch hits: {detail.get('prefetch_hits')})")
    print(f"  fixed budget holds fp16={detail.get('fp16_resident')} vs int4={detail.get('int4_resident')} layers")
    raise SystemExit(0 if ok else 1)

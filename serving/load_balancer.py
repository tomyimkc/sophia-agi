# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Cache-aware request router for a fleet of inference workers.

The naive load balancer (round-robin / least-loaded) is *cache-hostile*: it
scatters requests that share a prompt prefix across different workers, so each
worker re-prefills the shared prefix and the cluster-wide KV hit rate collapses.
Production routers (SGLang's cache-aware DP router, DeepSeek's prefix-routed
serving) instead use **prefix affinity**: send a request to the worker that
already holds the longest prefix of its tokens — *unless* that worker is hot,
in which case fall back to balance load.

This router implements exactly that policy:

1. **Affinity.** Each worker advertises which sealed prefix blocks it holds (a
   set of block hashes, fed by ``TieredKVCache.block_hashes``). For a new
   request the router scores every worker by how many leading blocks it can
   serve, and prefers the longest match — that prefill is skipped.

2. **Load cap.** Affinity is overridden when the best-cache worker's load
   exceeds ``balance_factor × mean_load`` (it's a hotspot). Then the request is
   routed to the least-loaded worker instead, trading a cache miss for
   tail-latency protection. This bounds load imbalance.

3. **Cold fallback.** With no prefix match anywhere, route by *consistent
   hashing* of the prefix, so identical future prompts deterministically land on
   the same worker and warm one cache rather than smearing across the fleet.

Dependency-free and deterministic. The router tracks the affinity registry
itself, so it models the cluster's cache state; ``record_completion`` updates a
worker's advertised blocks the way a real control plane would from heartbeats.

Falsifiable offline invariants (``offline_invariants()``):
  - a request is routed to the worker holding its prefix (affinity beats RR);
  - a hot worker is bypassed even when it has the best cache (load cap fires);
  - cold prefixes hash-route deterministically (same prefix → same worker);
  - the router never returns a worker id outside the fleet;
  - cache-aware routing yields a strictly higher cluster prefix-hit rate than
    round-robin on a prefix-skewed workload.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from serving.kv_cache import block_hashes


@dataclass
class _Worker:
    wid: int
    blocks: set[str] = field(default_factory=set)  # sealed prefix-block hashes held
    inflight: int = 0                               # current load (active requests)
    served: int = 0                                 # lifetime requests served


@dataclass
class RouteDecision:
    worker: int
    reason: str                 # "affinity" | "load_cap" | "cold_hash"
    prefix_blocks_hit: int      # leading blocks the chosen worker already holds
    best_affinity_worker: int   # who had the longest cache (may differ if capped)
    best_affinity_blocks: int


class CacheAwareRouter:
    """Routes requests across ``n_workers`` using prefix affinity + a load cap."""

    def __init__(
        self,
        n_workers: int,
        *,
        block_size: int = 16,
        balance_factor: float = 1.5,
    ) -> None:
        if n_workers <= 0:
            raise ValueError("n_workers must be positive")
        if balance_factor < 1.0:
            raise ValueError("balance_factor must be >= 1.0")
        self.block_size = block_size
        self.balance_factor = balance_factor
        self._workers = [_Worker(i) for i in range(n_workers)]
        self.route_reasons: dict[str, int] = {"affinity": 0, "load_cap": 0, "cold_hash": 0}

    @property
    def n_workers(self) -> int:
        return len(self._workers)

    # ---- affinity scoring --------------------------------------------------

    def _prefix_match(self, worker: _Worker, hashes: list[str]) -> int:
        """Leading blocks of ``hashes`` that ``worker`` holds (stops at first gap)."""
        n = 0
        for key in hashes:
            if key in worker.blocks:
                n += 1
            else:
                break
        return n

    def _consistent_hash(self, hashes: list[str]) -> int:
        seed = hashes[0] if hashes else "∅"
        h = hashlib.blake2b(seed.encode(), digest_size=8).digest()
        return int.from_bytes(h, "little") % self.n_workers

    def _mean_load(self) -> float:
        return sum(w.inflight for w in self._workers) / self.n_workers

    # ---- public API --------------------------------------------------------

    def route(self, token_ids: list[int]) -> RouteDecision:
        """Pick a worker for a request. Updates load; does not seal blocks yet."""
        hashes = block_hashes(token_ids, self.block_size)

        # Best affinity worker (ties → lower load, then lower id for determinism).
        best = max(
            self._workers,
            key=lambda w: (self._prefix_match(w, hashes), -w.inflight, -w.wid),
        )
        best_hit = self._prefix_match(best, hashes)

        if best_hit == 0:
            # Cold: nobody caches this prefix → deterministic hash placement.
            chosen = self._workers[self._consistent_hash(hashes)]
            reason = "cold_hash"
        else:
            mean = self._mean_load()
            cap = self.balance_factor * mean
            # Hotspot guard: if the best-cache worker is overloaded relative to
            # the fleet, divert to the least-loaded worker instead.
            if mean > 0 and best.inflight > cap and best.inflight > 0:
                chosen = min(self._workers, key=lambda w: (w.inflight, w.wid))
                reason = "load_cap"
            else:
                chosen = best
                reason = "affinity"

        chosen.inflight += 1
        chosen.served += 1
        self.route_reasons[reason] += 1
        chosen_hit = self._prefix_match(chosen, hashes)
        return RouteDecision(
            worker=chosen.wid,
            reason=reason,
            prefix_blocks_hit=chosen_hit,
            best_affinity_worker=best.wid,
            best_affinity_blocks=best_hit,
        )

    def record_completion(self, worker: int, token_ids: list[int]) -> None:
        """Mark a request done: free its load and seal its blocks on the worker.

        After a worker finishes prefill+decode it physically holds the request's
        prefix blocks, so future requests sharing that prefix should route to it.
        """
        w = self._workers[worker]
        w.inflight = max(0, w.inflight - 1)
        for key in block_hashes(token_ids, self.block_size):
            w.blocks.add(key)

    def load_snapshot(self) -> dict[int, int]:
        return {w.wid: w.inflight for w in self._workers}

    def served_snapshot(self) -> dict[int, int]:
        return {w.wid: w.served for w in self._workers}


# ---------------------------------------------------------------------------
# Offline invariants — deterministic, no deps, CI-gated.
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    bs = 4

    # 1. Affinity beats round-robin: warm worker 2, then a sharing request lands there.
    r = CacheAwareRouter(4, block_size=bs)
    prompt = list(range(16))  # 4 blocks
    d0 = r.route(prompt)
    r.record_completion(d0.worker, prompt)
    follow = prompt + [101, 102]  # shares all 4 prefix blocks
    d1 = r.route(follow)
    checks["affinity_routes_to_cache"] = (
        d1.worker == d0.worker and d1.reason == "affinity" and d1.prefix_blocks_hit == 4
    )

    # 2. Load cap fires: overload the cache-holding worker, next request diverts.
    r2 = CacheAwareRouter(3, block_size=bs, balance_factor=1.5)
    p = list(range(16))
    first = r2.route(p)
    r2.record_completion(first.worker, p)
    # pile inflight load onto that worker without completing (simulate hot worker)
    hot = r2._workers[first.worker]
    hot.inflight = 10
    d = r2.route(p)  # same prefix; best cache is the hot worker
    checks["load_cap_diverts_hotspot"] = (
        d.reason == "load_cap"
        and d.worker != first.worker
        and d.best_affinity_worker == first.worker
    )

    # 3. Cold prefixes hash deterministically (same prefix → same worker, repeatably).
    r3 = CacheAwareRouter(8, block_size=bs)
    cold = [7, 7, 7, 7, 8, 8, 8, 8]
    w_a = r3._consistent_hash(block_hashes(cold, bs))
    w_b = r3._consistent_hash(block_hashes(cold, bs))
    checks["cold_hash_deterministic"] = w_a == w_b

    # 4. Never route outside the fleet.
    r4 = CacheAwareRouter(5, block_size=bs)
    ids_ok = True
    for i in range(50):
        dec = r4.route([i % 9] * 8 + [i])
        if not (0 <= dec.worker < 5):
            ids_ok = False
        r4.record_completion(dec.worker, [i % 9] * 8 + [i])
    checks["routes_within_fleet"] = ids_ok

    # 5. Cache-aware > round-robin on a prefix-skewed workload (cluster hit rate).
    #    Build many requests over a few shared base prompts.
    bases = [list(range(0, 16)), list(range(100, 116)), list(range(200, 216))]
    workload = []
    for k in range(60):
        base = bases[k % len(bases)]
        workload.append(base + [900 + k])  # shared 4-block prefix + unique tail

    def cluster_hit_rate(router: CacheAwareRouter, round_robin: bool) -> float:
        total_blocks = 0
        hit_blocks = 0
        rr = 0
        for req in workload:
            hashes = block_hashes(req, bs)
            total_blocks += len(hashes)
            if round_robin:
                w = router._workers[rr % router.n_workers]
                rr += 1
                hit_blocks += router._prefix_match(w, hashes)
                w.inflight = max(0, w.inflight - 1)
                for key in hashes:
                    w.blocks.add(key)
            else:
                dec = router.route(req)
                hit_blocks += dec.prefix_blocks_hit
                router.record_completion(dec.worker, req)
        return hit_blocks / total_blocks if total_blocks else 0.0

    rr_rate = cluster_hit_rate(CacheAwareRouter(4, block_size=bs), round_robin=True)
    ca_rate = cluster_hit_rate(CacheAwareRouter(4, block_size=bs), round_robin=False)
    checks["cache_aware_beats_round_robin"] = ca_rate > rr_rate

    ok = all(checks.values())
    return ok, {
        "checks": checks,
        "round_robin_hit_rate": round(rr_rate, 3),
        "cache_aware_hit_rate": round(ca_rate, 3),
    }


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Load balancer offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  round-robin hit rate : {detail['round_robin_hit_rate']}")
    print(f"  cache-aware hit rate : {detail['cache_aware_hit_rate']}")
    raise SystemExit(0 if ok else 1)

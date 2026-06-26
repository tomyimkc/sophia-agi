<!-- SPDX-License-Identifier: Apache-2.0 -->
# Sophia storage workspace

Rust crates implementing the [distributed-storage roadmap](../docs/storage/STORAGE_ROADMAP.md),
mapping the repo toward the high-performance-storage skill set (Rust, async,
multithreading, durable I/O, consensus).

| crate | phase | what it is |
|---|---|---|
| [`kvcache`](#kvcache-phase-1) | 1 / 1b | sharded async in-memory KV cache + pipelining |
| [`diskstore`](#diskstore-phase-2) | 2 | bitcask-style durable engine; std + io_uring read backends |
| [`miniraft`](#miniraft-phase-3) | 3 | clean-room Raft core + deterministic fault-injection simulator |
| [`infcache`](#infcache-phase-4) | 4 | prefix-keyed, RAM→SSD tiered KV-block cache for LLM inference |

Run everything: `cargo test` (workspace root `storage/`).

---

## `kvcache` (Phase 1)

A sharded, async, in-memory **KV cache** in Rust, and the Python seam that lets
Sophia's RAG path offload hot reads to it.

> **Honest scope.** Single node, in-memory only, request/response (no
> pipelining), no persistence, no replication. It is a *cache*, not a database —
> losing it costs latency, never correctness. Persistence (io_uring on-disk
> engine) and replication (Raft) are Phases 2–3, deliberately not faked here.

## Architecture

```
            ┌──────────────────────── kvcache-server (Tokio) ────────────────────────┐
 client ───►│  accept loop ──► one async task per connection                          │
 (TCP)      │                      │                                                  │
            │                      ▼   route by FNV-1a(key) % N                       │
            │   ShardedCache: [ Mutex<Lru> ][ Mutex<Lru> ] ... N shards               │
            │                      │                                                  │
            │                      ▼   O(1) get/insert/evict, per-entry TTL           │
            │             intrusive doubly-linked-list LRU + slab                     │
            └──────────────────────────────────────────────────────────────────────-─┘
```

- **Sharding** (`cache.rs`) — keys hash (FNV-1a, deterministic across processes)
  to one of N independent shards, each an `Lru` behind its own `std::sync::Mutex`.
  Lock contention scales as ~1/N; the lock is never held across `.await`.
- **LRU with TTL** (`lru.rs`) — O(1) get/insert/evict via a slab + intrusive
  doubly-linked list; no scan to find the victim. TTLs expire lazily on access.
- **Wire protocol** (`protocol.rs`) — length-prefixed, big-endian, binary-safe
  (keys/values may contain any bytes). Ops: GET, SET (with TTL), DEL, PING, STATS.
- **Server** (`server.rs`) — Tokio, one task per connection, `TCP_NODELAY`,
  buffered reader/writer. Concurrency comes from many connections.

## Build, test, run

```bash
cd storage
cargo test                 # 13 unit + 5 integration tests, all offline
cargo clippy --all-targets # clean
cargo run --release --bin kvcache-server -- --addr 127.0.0.1:7070 --shards 16 --capacity 1000000
```

## Benchmark

`kvcache-bench` spawns the real server on an ephemeral port and drives concurrent
clients over loopback TCP — it measures the **full** client→TCP→server→client
round trip, not just the in-memory map.

```bash
cargo run --release --bin kvcache-bench -- --clients 32 --ops 30000 --keys 100000
```

Representative run (32 clients × 30k GETs, 100k keys, 256-byte values, 16 shards;
loopback, results vary by host — these are from CI-class hardware, recorded in
[../RESULTS.md](../RESULTS.md)):

| pipeline depth | throughput | p50 | p99 |
|---|---|---|---|
| 1 | ~186k ops/sec | ~168 µs/op | ~300 µs/op |
| 16 | ~1.60M ops/sec | ~306 µs/batch | ~546 µs/batch |
| 64 | ~2.13M ops/sec | ~920 µs/batch | ~1602 µs/batch |

Phase 1b added **pipelining** (`Client::pipeline`, `--pipeline DEPTH`): the client
batches requests into one flush and the server coalesces responses, trading
per-batch latency for ~8.6× throughput at depth 16. The depth-1 row is the honest
single-request baseline. These numbers exist to make later optimizations
measurable, not to claim they are good yet.

## Using it from Sophia (Python)

The cache is **opt-in and fail-closed**. With `SOPHIA_KVCACHE_ADDR` unset,
nothing changes. With it set, `agent/vector_store.py:search()` caches results
keyed by the query, `top_k`, and a fingerprint of the chunk set + query
embedding; a dead cache silently degrades to a normal search.

```bash
# terminal 1
cargo run --release --bin kvcache-server -- --addr 127.0.0.1:7070
# terminal 2
export SOPHIA_KVCACHE_ADDR=127.0.0.1:7070   # that's the only switch
```

`agent/kvcache_client.py` is a dependency-free pure-Python client speaking the
same protocol; its lenient `get`/`set` never raise, the strict `*_strict`
variants do.

---

## `diskstore` (Phase 2)

A bitcask-style **durable** KV engine: every write appends a CRC-checked record
to an append-only log and updates an in-memory keydir (`key → value offset`);
reads are one positional read. Crash recovery replays the log and truncates a
torn tail; `compact()` rewrites only live values.

```
put k,v  ──► append [crc|tstamp|klen|vlen|key|val] ──► keydir[k] = (offset, len)
get k    ──► keydir[k] ──► pread(value_offset, len)
recover  ──► scan log, last-writer-wins, stop at first bad CRC, truncate tail
```

The interesting part is `multi_get`, which routes batched reads through a
`BatchReader`:
- **`StdReader`** — one `pread` per key (portable, default, CI-tested).
- **`UringReader`** (feature `io_uring`) — pushes the whole batch into one
  io_uring submission and reaps completions; verified byte-identical to pread.

```bash
cargo test -p diskstore                                    # std path
cargo test -p diskstore --features io_uring                # + io_uring parity test
cargo run --release -p diskstore --features io_uring --bin diskstore-bench
```

Benchmark numbers and an **honest** note on when io_uring actually wins (it ties
pread on a page-cached set; the win needs real I/O) are in
[../RESULTS.md](../RESULTS.md).

---

## `miniraft` (Phase 3)

A **clean-room Raft** consensus core, built from the paper (not wrapped over a
library), plus a deterministic simulator that proves its safety properties under
faults. The node is a pure state machine — no I/O, no clock — so the driver
(`Sim`) can advance logical time, crash/restart nodes, and partition the network,
making the hard properties reproducibly testable:

```
Sim::new(5) ─► run_until(leader) ─► propose ─► replicate to quorum ─► commit
   │ crash(leader)  ─► remaining quorum re-elects, keeps committing
   │ partition([2],[3]) ─► minority CANNOT commit; majority does
   │ heal() ─► minority converges, uncommitted writes dropped
```

Implements election (randomized timeouts + up-to-date-log voting restriction),
log replication (consistency check + conflict truncation), and the commit rule
(quorum **and** current term). Snapshots, membership changes, and on-disk
persistence are explicitly out of scope (the node marks which state is
persistent; wiring it to `diskstore` is the next step).

```bash
cargo test -p miniraft     # 5 safety tests + doctest, fully deterministic
```

See [../RESULTS.md](../RESULTS.md) for the property → test table.

---

## `infcache` (Phase 4)

The role's core responsibility: a **prefix-keyed, tiered KV-block cache for LLM
inference**, composing the earlier phases. A token stream is chunked into blocks
whose keys hash the entire prefix, so a shared prompt prefix is a cache hit
(context caching). Blocks live in a RAM hot tier (`kvcache::ShardedCache`) backed
by a durable SSD tier (`diskstore::Bitcask`); an SSD hit is promoted back to RAM.

```
tokens ─► block_keys (prefix-chained) ─► plan_prefill
   reused prefix (cache hit)  │  recompute tail (miss)
   get_block: RAM ─miss─► SSD ─hit─► promote to RAM
   put_block: write-through to RAM + SSD (durable)
```

`plan_prefill` reports how much of a prompt is reusable vs. must be recomputed —
the number that drives inference cost. On a shared-system-prompt workload the
demo shows **94.1% prompt-token reuse**:

```bash
cargo test -p infcache
cargo run --release -p infcache --bin infcache-bench -- --requests 2000 --system 2048 --suffix 128
```

It's the storage/reuse layer, not an attention kernel (block payloads are opaque
serialized K/V). See [../RESULTS.md](../RESULTS.md) for numbers and scope.

## What's next (see the roadmap)

The four phases are shipped. Remaining productionization — per crate — is tracked
in the [roadmap](../docs/storage/STORAGE_ROADMAP.md): request pipelining tuning,
rolling segments + `O_DIRECT` for the engine, snapshots/membership/disk-persistence
for Raft, and integrating `infcache` under a real inference engine.

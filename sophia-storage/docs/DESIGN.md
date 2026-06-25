<!-- SPDX-License-Identifier: Apache-2.0 -->
# sophia-storage — Design

This document explains *why* the two crates are shaped the way they are, what is
real vs. a seam, and how each piece maps to the requirements of a
high-performance distributed-storage role (KVCache, distributed FS/object store,
Rust/C++, consensus, RDMA, io_uring/SPDK). It is written to be read alongside the
code — every claim points at a file.

## 0. First principles

Three commitments drive every decision here, in priority order:

1. **Correctness before speed.** A storage engine that loses or corrupts data
   silently is worthless no matter how fast. So the first thing each crate gets
   right is the failure boundary: CRC-framed records, torn-tail detection, WAL
   replay, ref-counted pins. The benchmarks come *after* the tests.
2. **Honest baselines, then optimize.** We measure the naive path (fsync per
   write, in-memory tiers) and publish the unflattering number, then point
   optimization at the measured bottleneck. No hand-wavy "it'll be fast."
3. **Algorithms separated from the substrate.** The LSM merge logic doesn't know
   whether bytes land on `std::fs` or `io_uring`; the KVCache placement logic
   doesn't know whether a tier is host DRAM or a remote RDMA pool. That
   separation is what lets the hard systems work (io_uring, CUDA, RDMA) drop in
   without rewriting the parts that are already correct.

These mirror the repo's own "measured-or-it-didn't-happen" culture
([RESULTS.md](../../RESULTS.md)).

---

## 1. Why these two crates, for *this* repo

Sophia is an LLM inference + retrieval + durable-write system. A storage engineer
earns their keep in exactly those three places, and two of them are acute here:

- **Inference (KVCache).** Sophia runs best-of-N sampling and multi-agent council
  deliberation (`agent/best_of.py`, `agent/council_deliberate.py`,
  `agent/sector_council.py`). These issue many requests over the *same* long
  prompt prefix — the textbook case for paged, prefix-shared KV reuse
  (PagedAttention / Mooncake). Without it, every sample re-prefills the whole
  prompt. The `sophia-kvcache` bench measures **97% of prefill avoided** on this
  pattern.
- **Durable writes (local engine).** Claims, decisions, preferences, and the task
  queue are append-only JSONL (`sophia_contract/stores.py`, `queue.py`). JSONL is
  hand-auditable — a genuine virtue for a trust layer — but every read scans the
  file and every "compaction" rewrites it whole. `sophia-lsm` keeps the same
  idempotent, durable, recoverable semantics on a log-structured engine.

The third place — a distributed FS / object store for training data — is a larger
build; it's sketched in §5 as the natural extension, not implemented here.

---

## 2. sophia-kvcache

### 2.1 Model

A **block** (`block.rs`) holds the KV tensors for a fixed run of tokens — the
paged unit, exactly like PagedAttention pages. Blocks are **content-addressed**:
`BlockId::derive(parent, tokens)` folds the parent block's hash into an FNV-1a
over the token ids, so

- identical prompt prefixes ⇒ identical block ids ⇒ **the KV is stored once**, and
- the *same* tokens after a *different* prefix ⇒ different ids (position-dependent,
  which attention requires).

The **prefix index** (`prefix.rs`) turns a token stream into its chain of block
ids and computes the longest contiguous resident prefix — the reuse length.

### 2.2 Tiering and eviction

`tier.rs` models three capacity-bounded arenas, **HBM → DRAM → NVMe**. On a hit a
block is promoted toward HBM; under pressure the LRU victim is demoted one tier
down, cascading to eviction only off the coldest tier.

`eviction.rs` is **reference-counted LRU**. This is the load-bearing invariant for
correctness under sharing: while a request's decode is in flight its prefix blocks
are *pinned* (refcount > 0) and cannot be evicted out from under it, even by a
flood of new prompts. The test `pinned_prefix_is_not_evicted_under_pressure`
hammers a 2-block tier with 40 distinct prompts and asserts the pinned prefix
survives.

### 2.3 What's real vs. seam

| Real (tested) | Seam (documented) |
|---|---|
| Content-addressed ids, prefix sharing | Physical tiers are in-memory maps |
| Promotion / demotion / cascade eviction | Transfer = `clone` today → `cudaMemcpyAsync` (HBM↔DRAM), RDMA read (DRAM↔remote), io_uring (DRAM↔NVMe) |
| Ref-counted pinning | Single node; disaggregated remote-DRAM pool is §4 |
| Hit-ratio / prefill-avoided metrics | Real KV tensor layout & dtype |

### 2.4 The RDMA / zero-copy seam (加分项)

The transfer methods in `tier.rs` are the only place bytes move between tiers.
The production path:

- **HBM↔DRAM:** `cudaMemcpyAsync` on a copy stream, overlapped with compute.
- **DRAM↔remote DRAM pool:** one-sided **RDMA** `READ`/`WRITE` verbs against
  registered, pinned buffers — the block payload is already the right shape to
  register once and reuse. This is the disaggregated KVCache topology (a shared
  remote memory pool the scheduler pulls warm prefixes from) and is where the
  "榨干现代网络硬件" bonus is won.
- **DRAM↔NVMe:** `io_uring` (or SPDK for userspace NVMe) batched reads.

None of these change a line of the placement/eviction logic above.

---

## 3. sophia-lsm

### 3.1 The write path

`put`/`delete` →
1. frame the record (`record.rs`: `[kind][klen][vlen][key][value][crc32]`),
2. append to the **WAL** and `fsync` (`wal.rs`) — the durability boundary,
3. apply to the **memtable** (`memtable.rs`, a sorted `BTreeMap`),
4. when the memtable passes its byte threshold, **flush** it to an immutable,
   sorted **SSTable** (`sstable.rs`) and truncate the WAL.

`get` checks memtable, then SSTables newest→oldest, stopping at the first table
that knows the key (value *or* tombstone). The SSTable has a CRC-checked **sparse
index** (one entry per 16 records) so a lookup binary-searches to a block and
scans a bounded window.

**Compaction** (`compaction.rs`) merges runs newest-wins and reaps tombstones,
collapsing N tables back to one.

### 3.2 Crash semantics (the part that must be right)

Framing is self-describing and CRC-guarded so a **torn write** after a crash is
*detectable*: on replay, a record that runs past EOF or fails CRC truncates the
log there — the last partial write is lost, everything before it survives. This
is at-least-once durability with a clean recovery boundary, which is exactly the
contract `queue.py` already promises ("survives restarts; safe to retry
everything"). Tested by `survives_reopen` (drop without flush, reopen, data is
there) and the `record` torn-tail / bit-flip tests.

### 3.3 What's real vs. seam

| Real (tested) | Seam (documented) |
|---|---|
| WAL append+fsync+replay | Bloom filter per SSTable (skip tables that can't hold the key) |
| **Group-commit `WriteBatch`** (one fsync per batch) | `O_DIRECT` + page-aligned registered buffers (true zero-copy) |
| **Real io_uring backend** (`io::IoUringIo`: Write/Read/Fsync SQEs) | SQPOLL (drop the submit syscall); concurrent commit thread |
| Memtable, SSTable write/read, sparse index | Leveled / size-tiered compaction to bound write-amp |
| Full-merge compaction + tombstone reap | Block cache for hot SSTable reads |
| CRC framing, torn-tail recovery | Concurrent readers / MVCC snapshot reads |
| Pluggable `IoBackend` trait | |

### 3.4 What the benchmark said, and what we did about it

The first measurement:

```
put+fsync   1,212 ops/s   p50 0.71 ms      get   3.56M ops/s   p50 128 ns
```

The read path is already memory-fast. The write path was **fsync-per-write
bound** — every `put` forced the platter. So we built the two things that single
number pointed at:

- **Group commit (`WriteBatch`, `wal.rs::append_batch`):** N records, one
  `fsync`. Same durability; the platter sync is amortized across the batch.
  Measured:

  ```
  batch=1        1,164 writes/s    1.0x
  batch=8        8,639 writes/s    7.1x
  batch=64      55,491 writes/s   45.8x
  batch=512    247,053 writes/s  203.8x   (one fsync per 512 writes)
  ```

- **io_uring backend (`io::IoUringIo`):** each batched record becomes a `Write`
  SQE submitted in a single `io_uring_enter`, followed by one `Fsync` SQE — so
  the append side of the group commit collapses to one submission too, and reads
  go through `Read` SQEs. Exercised end-to-end (`io_uring_backend_round_trips`):
  the same engine — WAL replay, SSTable write+read, group-commit batch — runs on
  the ring. `O_DIRECT` + registered buffers and SQPOLL are the documented next
  steps for true zero-copy.

This is the RocksDB design lineage the JD names, and the point of building it
this way is to show the reflex: measure, find the real bottleneck, fix *that*.

---

## 4. Distributed durability — Raft (consensus 加分项)

The queue/decision log is honestly single-process today (VISION.md flags "a
concurrent queue" as deferred). The HA design:

- A **Raft** group (3–5 nodes) where the replicated log *is* the decision/queue
  log. Each committed entry is a framed record applied to a local `sophia-lsm`
  state machine; SSTable flushes double as the snapshot mechanism for log
  compaction.
- **Linearizable reads** via leader leases; idempotency keys (already in
  `queue.py`) make at-least-once delivery safe across leader changes.
- Built on `openraft`/`raft-rs` — the `sophia-lsm` engine slots in directly as
  the Raft log + state-machine store.

This is where "分布式事务、Paxos/Raft 等共识机制" becomes demonstrable rather
than asserted.

## 5. Distributed FS / object store (the larger build)

The natural extension for "下一代分布式文件系统和对象存储系统": expose Sophia's
provenance corpus and the OKF belief graph as a **content-addressed object
store** — blocks are immutable and hash-named (the KVCache already works this
way), so dedup and integrity are free, and a FUSE front-end (another bonus line)
can present the belief graph as a read-mostly filesystem. Erasure coding +
placement across nodes is the bandwidth/durability story for training data. Not
built here; listed so the trajectory from these two crates to the full JD scope
is explicit.

---

## 6. Non-goals / guardrails

- **No coupling to the Python trust layer.** This workspace is opt-in and
  feature-gated; JSONL stays Sophia's auditable default. The fast engine is a
  path you choose, never the substrate that decides whether Sophia is checkable.
- **No fake speed.** Anything not yet measured is labeled a seam. The in-memory
  tiers and `clone`-based transfers are not pretending to be RDMA.
- **No dependency sprawl.** std-only until a seam is genuinely built out, so the
  thing always clones and runs.

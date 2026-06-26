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

### 2.3 Storage media behind a tier

Each tier is an `Arena` over a pluggable `BlockStore` (`store.rs`), so the
placement logic is identical regardless of medium:

- **HBM / DRAM → `MemStore`** (in-memory). A GPU build swaps in a CUDA
  allocation; the read/write becomes `cudaMemcpyAsync`. **No GPU is needed to
  run or test the controller** — that is the point of the split.
- **NVMe → `FileStore`** — **real on-disk persistence**, one file per block
  (`[token_count][payload]`, temp-write+rename so a reader never sees a torn
  block), with an in-memory id index so `contains`/`ids`/`len` never touch the
  disk. Demoted blocks survive, page back in on promotion, and persist across a
  cache restart (the index rebuilds from the directory). Tested by
  `nvme_tier_persists_demoted_blocks_to_disk` (a block is pushed down to NVMe,
  asserted present as a file on disk, then promoted back without recompute) and
  `store::file_store_persists_to_disk_and_survives_reopen`.

Every store tracks cumulative `bytes_in`/`bytes_out`, so the cache reports the
volume crossing the slow boundary (`KvCache::nvme_bytes`) — the number a
zero-copy / RDMA / SPDK path exists to shrink. The bench measured **2.3 MiB
written / 2.2 MiB read back** on the disk tier under the council workload.

#### What's real vs. seam

| Real (tested) | Seam (documented) |
|---|---|
| Content-addressed ids, prefix sharing | HBM/DRAM are in-memory (no GPU here) → CUDA alloc + `cudaMemcpyAsync` |
| **Real disk NVMe tier** (`FileStore`), persists + reloads | NVMe transfer is `write`/`read` → `io_uring`/SPDK |
| Promotion / demotion / cascade eviction across media | Disaggregated remote-DRAM pool over RDMA is §4 |
| Ref-counted pinning; byte-movement accounting | Real KV tensor layout & dtype; per-block CRC |
| Hit-ratio / prefill-avoided / NVMe-bytes metrics | |

### 2.4 The RDMA / zero-copy seam (加分项)

A tier transfer is exactly a `store.take` on the source + `store.put` on the
destination (`tier.rs` / `store.rs`) — the only place bytes move between tiers.
The production path swaps the medium behind those two calls:

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

`get` checks memtable, then tables in level read-order (L0 newest→oldest, then
L1, L2, …), stopping at the first table that knows the key (value *or*
tombstone). Each SSTable carries a **bloom filter** (`bloom.rs`, ~1% FP, double
hashing) checked first — a definite miss skips the table with *zero* I/O — and a
CRC-checked **sparse index** (one entry per 16 records) that narrows a possible
hit to a bounded scan.

**Leveled compaction** (`levels.rs` + `Engine::maybe_compact`) bounds write
amplification. A flush lands in **L0** (a few possibly-overlapping tables); when
L0 reaches `compaction_trigger` it merges into **L1**; each deeper level holds a
single sorted run `FANOUT`× larger than the one above and is rewritten only when
it overflows its record budget, cascading downward. A key is rewritten
O(levels) times, not O(dataset) as a full merge would. Tombstones are reaped only
when merging into the **deepest** level (nothing below can resurrect the key). A
small text **manifest** (temp-write+rename) records which table id sits at which
level so the layout survives a restart; SSTables absent from the manifest are
orphans from an interrupted compaction and are ignored on open. Tested by
`leveled_compaction_cascades_and_stays_correct` (forces L0→L1→L2, checks reads +
tombstone survival) and `survives_reopen_after_compaction`.

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
| WAL append+fsync+replay | `O_DIRECT` + page-aligned registered buffers (true zero-copy) |
| **Group-commit `WriteBatch`** (one fsync per batch) | SQPOLL (drop the submit syscall); concurrent commit thread |
| **Real io_uring backend** (`io::IoUringIo`: Write/Read/Fsync SQEs) | Block cache for hot SSTable reads |
| Memtable, SSTable write/read, sparse index | Concurrent readers / MVCC snapshot reads |
| **Bloom filter per SSTable** (skip definite misses, no I/O) | Per-level key-range partitioning (many runs per level) |
| **Leveled compaction** (L0→L1→…, manifest, budget cascade) | |
| Tombstone reap (only at the deepest level) | |
| CRC framing, torn-tail recovery; pluggable `IoBackend` | |

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

## 4. Distributed durability — Raft (consensus 加分项) — implemented

The queue/decision log is single-process JSONL today (VISION.md flags "a
concurrent queue" as deferred). `sophia-raft` is the HA core that fixes it, built
from first principles rather than wrapping a library so the safety reasoning is
visible and tested.

- **`node::RaftNode`** is the algorithm, driven by explicit `tick()`/`step()` —
  no threads, no wall-clock — so the cluster is deterministic. It implements the
  safety-critical parts: leader election with the up-to-date-log vote restriction
  (§5.4.1), AppendEntries consistency check + conflict truncation (§5.3), and
  current-term-only commit advancement (§5.4.2).
- **`cluster::Cluster`** is an in-memory harness routing messages and driving
  logical time, with crash/restart/partition controls. It's the test substrate;
  production swaps it for a real transport (TCP/gRPC) and a durable per-node log
  — **the `sophia-lsm` engine slots straight in as the Raft log + snapshot
  store**, and idempotency keys (already in `queue.py`) make at-least-once
  delivery safe across leader changes.
- **`state_machine::StateMachine`** turns the committed stream into state; the
  reference `KvStateMachine` mirrors the decision-log / queue shape.

What is *proven*, deterministically (`lib.rs` tests): a 3-node cluster elects
exactly one leader; committed entries replicate to all nodes; a leader crash is
survived and committed data preserved while the survivors elect a new leader; a
2-of-5 **minority cannot commit**; and a partitioned follower **catches up** on
rejoin. That is "分布式事务、Paxos/Raft 等共识机制" demonstrable, not asserted.

Remaining (documented): real network transport, durable log on `sophia-lsm`,
snapshotting/log-compaction, membership changes, and leader-lease linearizable
reads.

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

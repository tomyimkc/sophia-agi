<!-- SPDX-License-Identifier: Apache-2.0 -->
# sophia-storage

> An **optional, isolated Rust workspace** for the high-performance storage layer
> beneath Sophia's Python trust layer. Nothing in the Python codebase imports it;
> it builds, tests, and benchmarks on its own.

This crate exists for two reasons at once:

1. **For Sophia** — pay down the real costs of the JSONL-everywhere persistence
   (`sophia_contract/stores.py`, `queue.py`) and make the verification /
   deliberation loop cheap enough to run fail-closed at scale.
2. **As a distributed-storage portfolio** — it implements, from first principles,
   the primitives a high-performance storage role asks for (KVCache for LLM
   inference, an LSM local engine, crash-checked durability), as runnable,
   tested, benchmarked code rather than a résumé bullet.

It is honest about being a **skeleton**: the data paths are correct and tested,
and the performance levers (io_uring, RDMA, bloom filters, leveled compaction,
consensus replication) are present as *documented seams*, not half-built
features. See [`docs/DESIGN.md`](docs/DESIGN.md) and the roadmap below.

## Crates

| Crate | What it is | JD line it targets |
|-------|------------|--------------------|
| [`sophia-kvcache`](crates/sophia-kvcache) | Disaggregated, prefix-sharing KV-cache: paged content-addressed blocks, HBM→DRAM→**real-disk NVMe** tiering, ref-counted LRU eviction, byte-movement accounting | 高性能 KVCache 存储系统; RDMA 零拷贝 (seam) |
| [`sophia-lsm`](crates/sophia-lsm) | Log-structured local engine: WAL + memtable + SSTable + bloom filters + leveled compaction, group commit, pluggable I/O backend (real io_uring), CRC-framed crash recovery | SSD 本地存储引擎; io_uring; RocksDB 设计范式 |
| [`sophia-raft`](crates/sophia-raft) | Deterministic Raft consensus core: leader election, log replication, current-term commit rule, in-memory cluster harness (election / crash / partition / catch-up) | 分布式事务; Paxos/Raft 共识机制 |

## Build & test

```bash
cd sophia-storage
cargo test --workspace                      # 27 tests (std backend; incl. real-disk NVMe tier)
cargo test -p sophia-lsm --features io_uring # +1 test through the real ring (Linux 5.1+)
cargo bench -p sophia-lsm                   # put/get latency + group-commit scaling
cargo bench -p sophia-kvcache               # prefix hit-ratio + prefill avoided + NVMe bytes
```

The **default** build is std-only and offline — it clones and runs anywhere. The
`io_uring` feature is the one place an external dependency (`io-uring`) and a real
kernel facility enter; the tier abstractions are where CUDA / RDMA verbs plug in
later, without touching the algorithms.

## Measured today (4-core container, see benches/)

```
sophia-lsm:
  put+fsync   1,212 ops/s   p50 0.71 ms   p99 2.70 ms   (fsync-per-write floor)
  get         3,563,013 ops/s   p50 128 ns   p99 679 ns  (memtable hits)

  group commit (one fsync per batch) — lifting the fsync-per-write floor:
    batch=1        1,164 writes/s    1.0x
    batch=8        8,639 writes/s    7.1x
    batch=64      55,491 writes/s   45.8x
    batch=512    247,053 writes/s  203.8x

sophia-kvcache  (512-tok prompt, fan-out 16, 200 rounds — the council/best-of-N shape):
  avg prefix hit-ratio   0.970
  prefill blocks avoided 97.0%   (105,600 → 3,200 blocks computed)
  NVMe tier (real disk)  2.3 MiB written / 2.2 MiB read back  (the RDMA/SPDK target)

sophia-kvcache GPU HBM tier (cuda feature) — measured on a real RunPod RTX 4090 (24 GiB):
  512 MiB round-tripped through HBM, 2048/2048 blocks byte-verified, RESULT: PASS
  H2D 8.91 GiB/s   D2H 4.99 GiB/s
  (single-stream, synchronous per-block, pageable host memory — a correctness +
   floor-bandwidth run; pinned buffers + cudaMemcpyAsync overlap is the next step)
```

We published the unflattering per-write number first, identified the fsync as the
bottleneck, then built group commit to amortize it — **204× at batch 512**, with
identical durability. Honest baseline, then optimize at the measured bottleneck —
the same discipline as the rest of the repo ([RESULTS.md](../RESULTS.md)).

## Roadmap (priority order)

1. ~~**LSM group-commit + io_uring backend**~~ — ✅ done: `WriteBatch` one-fsync
   group commit (204× at batch 512) and a real `io::IoUringIo` submitting
   Write/Read/Fsync SQEs, exercised end-to-end through the engine.
2. **KVCache tiers** — ◐ mostly done: `tier::Arena` is pluggable over a
   `BlockStore`; the **NVMe tier is real disk** (`store::FileStore`) and the
   **HBM tier is real GPU memory** (`store::CudaHbmStore`, feature `cuda`, real
   `cudaMemcpy`), exercised on a RunPod GPU box via
   `.github/workflows/gpu-kvcache.yml`. Remaining: a remote-DRAM pool over RDMA
   *(needs RDMA-NIC hardware)*.
3. **O_DIRECT + registered buffers + SQPOLL** on the io_uring backend; concurrent
   group commit (a commit thread coalescing per-op sync requests); `io_uring` on
   the NVMe `FileStore`.
4. ~~**Bloom filters + leveled compaction**~~ — ✅ done: a per-SSTable bloom
   filter (skips definite misses with no I/O) and leveled compaction
   (`levels.rs`: L0→L1→… with a manifest and per-level budgets) that bounds
   write amplification to O(levels) rewrites per key.
5. ~~**Raft-replicated log**~~ — ✅ done (`sophia-raft`): election, replication,
   current-term commit safety, leader-crash failover, minority-cannot-commit,
   follower catch-up — all deterministically tested. Production wiring (real
   transport + durable per-node log on `sophia-lsm`) is the remaining step.
6. **Python binding** — expose `sophia-lsm` under a `SOPHIA_STORAGE_ENGINE=lsm`
   flag behind the existing store interface; JSONL stays the default.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the rationale and the disaggregated /
consensus designs in full.

## Relationship to Sophia's charter

[VISION.md](../VISION.md) says: *"Assemble and orchestrate; innovate at the trust
layer. Don't try to out-train frontier labs."* A bespoke storage engine could be
scope creep, so this lives in an **isolated, feature-gated crate**: the Python
trust layer never depends on it, JSONL remains the auditable default, and this is
the optional fast path you opt into — not a rewrite of the substrate that makes
Sophia checkable.

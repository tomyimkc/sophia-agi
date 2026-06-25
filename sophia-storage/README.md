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
| [`sophia-kvcache`](crates/sophia-kvcache) | Disaggregated, prefix-sharing KV-cache: paged content-addressed blocks, HBM→DRAM→NVMe tiering, ref-counted LRU eviction | 高性能 KVCache 存储系统; RDMA 零拷贝 (seam) |
| [`sophia-lsm`](crates/sophia-lsm) | Log-structured local engine: WAL + memtable + SSTable + compaction, pluggable I/O backend, CRC-framed crash recovery | SSD 本地存储引擎; io_uring (seam); RocksDB 设计范式 |

## Build & test (zero external dependencies)

```bash
cd sophia-storage
cargo test --workspace        # 22 tests: framing, recovery, compaction, prefix reuse, eviction
cargo bench -p sophia-lsm     # put/get latency + ops/sec
cargo bench -p sophia-kvcache # prefix hit-ratio + prefill avoided
```

The whole workspace is **std-only** on purpose, so it clones and runs anywhere
with no network — the I/O-backend and tier abstractions are where real
dependencies (`io-uring`, CUDA, RDMA verbs) plug in later without touching the
algorithms.

## Measured today (4-core container, see benches/)

```
sophia-lsm:
  put+fsync   1,111 ops/s   p50 0.79 ms   p99 2.99 ms   (fsync-per-write bound)
  get         3,355,089 ops/s   p50 128 ns   p99 820 ns  (memtable hits)

sophia-kvcache  (512-tok prompt, fan-out 16, 200 rounds — the council/best-of-N shape):
  avg prefix hit-ratio   0.970
  prefill blocks avoided 97.0%   (105,600 → 3,200 blocks computed)
```

The put number is deliberately unflattering: it fsyncs on **every** write, which
is precisely the motivation for the group-commit + io_uring work in the roadmap.
Honest baselines first, optimization second — the same measurement discipline as
the rest of the repo ([RESULTS.md](../RESULTS.md)).

## Roadmap (priority order)

1. **KVCache zero-copy tiers** — back `tier::Arena` with real HBM/DRAM/NVMe;
   `cudaMemcpyAsync` + RDMA reads for the transfer path. *(biggest inference win)*
2. **LSM io_uring backend** — implement `io::IoUringIo`; group-commit the WAL.
3. **Bloom filters + leveled compaction** — bound read- and write-amplification.
4. **Raft-replicated log** — replicate the decision log / task queue for HA.
5. **Python binding** — expose `sophia-lsm` under a `SOPHIA_STORAGE_ENGINE=lsm`
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

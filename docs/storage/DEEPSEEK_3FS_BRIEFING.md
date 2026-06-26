# DeepSeek 3FS — interview briefing

A grounded study of DeepSeek's **Fire-Flyer File System (3FS)** for the
高性能分布式存储工程师 role, with an honest mapping to the storage work in this
repo (`storage/`). Sourced primarily from the official design notes in the
`deepseek-ai/3FS` repository; performance figures from DeepSeek's release
materials and secondary coverage are flagged as such.

> **Why read this before interviewing.** The job is on DeepSeek's storage team,
> and 3FS + its KVCache use case *is* the product. Knowing its architecture and
> the reasoning behind its choices lets you speak their vocabulary (CRAQ, chunk
> store, USRBIO, Iov/Ior) and connect your own work to it. The repo's storage
> phases were built deliberately along the same axes.

---

## 1. What 3FS is, and the problem it solves

3FS is a distributed file system purpose-built for large-scale AI training and
inference I/O. It pools thousands of NVMe SSDs across hundreds of storage nodes
into a single namespace, served over an RDMA network (InfiniBand / RoCE). Design
priorities: **massive aggregate bandwidth**, **strong consistency**, and a
**POSIX-ish file interface** so data loaders and KV caches integrate without a
bespoke API.

Reported performance (DeepSeek release / coverage — not in the design notes, treat
as vendor figures): ~**6.6 TiB/s** aggregate read on a 180-node / 200 Gbps IB
cluster; ~**40 GiB/s** peak read per client node for the LLM-inference KVCache
workload.

## 2. Architecture

Four decoupled components (from the design notes):

| Component | Role | Key choices |
|---|---|---|
| **Cluster Manager** | membership, config distribution, heartbeats | several managers, one primary; config in a coordination service (ZooKeeper/etcd-class) |
| **Metadata Service** | file-system semantics (open/create/listdir/rename) | **stateless**; all state in **FoundationDB** (transactional KV, serializable snapshot isolation). Clients hit any instance |
| **Storage Service** | chunk store on local SSDs | **CRAQ** chain replication; multiple storage targets per SSD, each in a different chain |
| **Client** | FUSE *or* native | FUSE for compatibility; **native USRBIO** for zero-copy async performance |

### Metadata on FoundationDB
Inodes (`INOD` prefix, monotonic 64-bit ids) and directory entries (`DENT`
prefix, parent-inode + name → contiguous ranges for fast listing) are KV pairs.
Read-only txns for `fstat`/`lookup`/`listdir`; read-write txns with automatic
conflict detection + retry for `create`/`rename`/`unlink`. Making metadata
*stateless over FDB* is the key scaling decision — it pushes the hard
distributed-transaction problem down to a system designed for it, rather than
reinventing it.

### Storage: chunk store + CRAQ
- **Chunk store:** physical files in 11 power-of-two sizes (64 KiB → 64 MiB), 256
  files per size category, allocation tracked by bitmaps; `fallocate()` fights
  fragmentation. Copy-on-write for updates, append optimization for in-place
  growth. **Chunk metadata is cached in memory and updated atomically in
  RocksDB** via write batches.
- **CRAQ (Chain Replication with Apportioned Queries):** write-all / read-any.
  Writes flow head→tail creating a pending version; the tail commits and the ack
  propagates back. **Reads can hit any chain member** (returning the committed
  version, or signaling a pending one) — that's what unlocks SSD+RDMA read
  throughput. Multiple targets per SSD join *different* chains to balance load.

### Client: FUSE vs native (USRBIO) — the io_uring connection
The design notes are blunt about FUSE's ceiling: memory copies across the
kernel/user boundary, a kernel-space spin lock that burns CPU, **~400K 4 KiB
reads/sec** before it stops scaling, and no concurrent writes to one file on
Linux 5.x. So 3FS ships a **native client (USRBIO)** explicitly *"inspired by
Linux io_uring"*:
- **Iov** — a large shared-memory region for I/O buffers, registered with the NIC
  (InfiniBand) for zero-copy.
- **Ior** — a small shared **ring buffer** for request/completion, exactly the
  io_uring submission/completion model.
- Hybrid: the FUSE daemon still handles metadata (open/close/stat); apps register
  fds and do the actual I/O through the native ring.

Writes pull data via **RDMA Read** (the storage node reads from the client's
registered memory), so the data path is zero-copy and CPU-light.

## 3. The design decisions worth being able to defend

These are the "first principles, not gluing libraries" points the JD prizes:
1. **CRAQ instead of Raft for the data plane.** Raft funnels reads through the
   leader; CRAQ's read-any across the chain converts every replica into read
   bandwidth — the right call when the workload is read-heavy and SSDs+RDMA have
   bandwidth to spare. (Raft-class consensus still fits a *control/metadata*
   plane — cf. FDB's internals.)
2. **Stateless metadata over a transactional KV (FDB)** rather than a hand-rolled
   distributed metadata store: reuse a system built for serializable
   transactions; scale metadata nodes horizontally because they hold no state.
3. **A custom io_uring-style user-space I/O path** because FUSE's copies and lock
   contention cap throughput far below what NVMe+RDMA can deliver.
4. **Chunk store with size-classed physical files + RocksDB chunk metadata**:
   bound fragmentation, keep allocation O(1)-ish, make metadata updates atomic.

## 4. How this repo's `storage/` work maps to 3FS

Honest correspondence — these are *learning-scale* analogues, not claims of parity:

| 3FS mechanism | Closest artifact here | Relationship |
|---|---|---|
| USRBIO Ior/Iov (io_uring-style ring) | `diskstore` io_uring batched reads + O_DIRECT bench | Same primitive. The O_DIRECT bench shows the *why*: io_uring 10.6× pread under real device I/O (`RESULTS.md`) |
| CRAQ chain replication (data plane) | `miniraft` (Raft) + `raftkv` durable state | Both solve replication/consistency. I built Raft from the paper; the briefing's point #1 is exactly the CRAQ-vs-Raft tradeoff — be ready to discuss it |
| Stateless metadata over FoundationDB | — (not built) | The honest gap. FDB's deterministic-simulation testing is the model my `miniraft` simulator gestures at |
| Chunk store on SSD + RocksDB metadata | `diskstore` (bitcask: append log + in-mem keydir + CRC recovery + compaction) | Same family (log-structured local engine); 3FS uses size-classed chunks + RocksDB, I use a single-file bitcask |
| KVCache for inference (cost-effective vs DRAM) | `infcache` (prefix-keyed, RAM→SSD tiered KV blocks) | Directly analogous: cache LLM context KV on a fast tier backed by SSD. My demo: 94.1% prompt-token reuse |

**Talking point:** "I built the same primitives at study scale and measured the
tradeoffs first-hand — io_uring's real win is cold/O_DIRECT I/O, Raft trades read
throughput for simplicity vs CRAQ, and prefix-keyed tiering is what makes KVCache
cheaper than pure DRAM. The gap to 3FS is the stateless-metadata-over-FDB plane
and production scale/RDMA."

## 5. Likely interview questions (and where to go deeper)

- *Why CRAQ over Raft/Paxos for the storage chains?* (read-any throughput; the
  cost is write latency through the chain). Read the CRAQ paper (van Renesse &
  Schneider, OSDI'04... actually Terrace & Freedman, USENIX ATC'09).
- *Why is FUSE slow, and how does USRBIO fix it?* (copies + spin-lock; ring +
  registered memory + RDMA Read). Read *To FUSE or Not to FUSE* (FAST'17) and the
  io_uring docs.
- *How does the chunk store avoid fragmentation / make metadata durable?*
  (size classes, bitmaps, `fallocate`, RocksDB write batches).
- *Why FoundationDB for metadata?* (serializable txns, proven simulation testing).
  Read the FoundationDB paper (SIGMOD'21).
- *How would you cache LLM KV for inference cost-effectively?* (prefix/blockwise
  keying, RAM→SSD tiering, RDMA fetch) — this is `infcache`'s exact territory.

**Before the interview:** clone `deepseek-ai/3FS`, read `docs/design_notes.md`
end-to-end, skim the storage service and USRBIO client code, and run their
benchmark scripts if you have hardware. Pair it with the CRAQ, FoundationDB, and
io_uring papers above.

## Sources

- DeepSeek 3FS design notes — https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md
- DeepSeek 3FS repository / README — https://github.com/deepseek-ai/3FS
- MarkTechPost overview (perf figures) — https://www.marktechpost.com/2025/02/28/deepseek-ai-releases-fire-flyer-file-system-3fs-a-high-performance-distributed-file-system-designed-to-address-the-challenges-of-ai-training-and-inference-workload/
- Phoronix coverage — https://www.phoronix.com/news/DeekSeek-3FS-File-System
- CRAQ: *Object Storage on CRAQ* (Terrace & Freedman, USENIX ATC'09)
- FoundationDB (Zhou et al., SIGMOD'21); *Efficient IO with io_uring* (Axboe);
  *To FUSE or Not to FUSE* (Vangoor et al., FAST'17)

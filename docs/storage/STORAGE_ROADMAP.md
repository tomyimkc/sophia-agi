# Distributed Storage Roadmap — targeting the DeepSeek 高性能分布式存储工程师 role

**Purpose.** A concrete, honest build + learning plan that turns demonstrable
distributed-storage engineering skill into artifacts living in (or beside) this
repo. It maps every line of the job description to a milestone, an
implementation, and the literature you should internalize.

> **Reality check.** Sophia today is a Python *reasoning/verification* platform.
> Its entire "storage layer" is `rag/index/chunks.jsonl`, an in-memory
> `embeddings.npz`, and a 44-line append-only JSONL decision log
> (`agent/memory.py`). There is **no Rust/C++, no I/O engine, no concurrency, no
> consensus**. This roadmap does not pretend otherwise. The strategy is to build
> a *real* storage subsystem that Sophia genuinely consumes — a credible,
> benchmarkable artifact, not a refactor of the AI code.

---

## 0. The one authentic bridge

The role's #1 responsibility is a **KVCache store for LLM inference**. Sophia
runs LLM inference and RAG retrieval — the *same* problem domain. So the work
plugs in naturally:

- `agent/vector_store.py` / `agent/retrieval.py` → backed by a real cache tier
  instead of an in-memory numpy array.
- LLM inference KV-block reuse → a prefix-keyed KVCache server (DeepSeek's actual
  production problem: Context Caching / disk-tiered KV).

The killer demo to aim for: **Sophia's RAG p99 retrieval latency drops measurably
when backed by the Rust cache tier vs. the Python in-memory path**, with numbers
in `RESULTS.md`.

---

## 1. Requirement → milestone → literature map

| Job text | What to build | Skill proven | Read first |
|---|---|---|---|
| 扎实的 Rust/C++、多线程、异步、性能分析 | Sharded async KV cache server (Tokio) + client; flamegraph/`perf` profiling | Rust, async, concurrency, profiling | *Rust Atomics and Locks* (Bos); Tokio docs; `tokio-console` |
| 支撑大模型推理的高性能 KVCache 存储系统，毫秒级延迟、上亿级 IOPS | Prefix-keyed KV-block cache, tiered RAM→SSD, admission/eviction | Latency/throughput engineering | DeepSeek Context Caching notes; Mooncake (KVCache-centric) paper |
| 下一代分布式文件系统/对象存储；强一致共享存储层 | Object store w/ Raft-replicated metadata + chunked data plane | Distributed systems, strong consistency | GFS; Ceph/RADOS; Tectonic (Meta, FAST'21); 3FS (DeepSeek's own FS) |
| 分布式事务、Paxos/Raft 共识 | `openraft` cluster (3 nodes): election, log, membership, snapshots | Consensus, replication | Raft paper (Ongaro); Paxos Made Simple; *FoundationDB* (SIGMOD'21) |
| 不简单套用 RocksDB/FoundationDB/ClickHouse，从第一性原理 | Your own LSM/bitcask engine + a written tradeoff analysis vs RocksDB | Design judgment | RocksDB wiki; LSM survey (Luo & Carey); WiscKey (FAST'16) |
| 加分: 零拷贝、RDMA、榨干网络硬件 | `mmap` zero-copy reads; RDMA transport for shard fetch (`async-rdma`) | Kernel-bypass networking | RDMA Aware Programming Manual; FaRM (NSDI'14); Mooncake transfer engine |
| 加分: SSD/HDD 本地引擎，io_uring/SPDK | On-disk engine on `io_uring` (`tokio-uring`/`glommio`); SPDK stretch | Storage-hardware-level I/O | *Efficient IO with io_uring* (Axboe); SPDK docs; *Understanding Modern Storage APIs* (ATC'22) |
| 加分: 内核文件系统 / FUSE 性能 | FUSE passthrough fs exposing the object store; profile the I/O stack | Kernel I/O stack | *Linux Programming Interface* (Kerrisk) ch. on FS; `libfuse`; To FUSE or Not (FAST'17) |
| 加分: OSDI/FAST/VLDB 论文或同等工程影响力 | Open-source the engine + a rigorous benchmark writeup | Research-grade rigor | Read 3–5 FAST/OSDI papers/yr; reproduce one |

---

## 2. Skills-gap self-assessment (be brutally honest with yourself)

Rate yourself 0–3 on each. The job assumes ~2+ on the core rows.

**Core (required):**
- [ ] Rust ownership/lifetimes/traits fluency
- [ ] `async`/await mental model + executor internals (Tokio)
- [ ] Lock-free / atomics / memory ordering
- [ ] Profiling: `perf`, flamegraphs, `tokio-console`, cachegrind
- [ ] Distributed systems fundamentals (CAP, linearizability, consensus)
- [ ] Raft implemented (not just read about)

**Adjacent (strongly expected):**
- [ ] LSM / B-tree internals; write/read/space amplification tradeoffs
- [ ] Page cache, fsync semantics, crash consistency, WAL
- [ ] Networking: TCP tuning, zero-copy (`sendfile`/`splice`/`mmap`)

**Bonus (differentiators):**
- [ ] `io_uring` programming; SPDK/NVMe
- [ ] RDMA verbs / one-sided ops
- [ ] FUSE / kernel FS internals
- [ ] A published paper or a widely-used OSS storage component

Wherever you're at 0–1 on a **Core** row, that's your next month.

---

## 3. Phased build plan (artifacts, not just reading)

Each phase ends with a runnable artifact + benchmark numbers committed to the repo.

### Phase 1 — Single-node KV cache server (weeks 1–4) → proves Rust + async
- `storage/` Cargo workspace; `kvcache-server` (Tokio) + `kvcache-client`.
- Sharded in-memory map, consistent hashing, binary or gRPC (`tonic`) protocol.
- Wire `agent/vector_store.py` to *optionally* fetch from it (feature-flagged,
  fail-closed to the existing in-memory path — consistent with Sophia's ethos).
- **Deliverable:** `cargo bench` + a `criterion` report; p50/p99 vs Python path.

### Phase 2 — On-disk engine with io_uring (weeks 5–10) → proves storage-hardware skill
- Bitcask-style append log + in-memory keydir, then a tiny leveled LSM.
- I/O via `tokio-uring`/`glommio`; crash-consistent WAL + fsync discipline.
- Benchmark IOPS/latency vs a `std::fs` baseline **and** a RocksDB baseline;
  write the *first-principles* tradeoff analysis (this is the 不简单套用 point).
- **Deliverable:** `STORAGE_ENGINE.md` design doc + reproducible benchmark.

### Phase 3 — Replication & consensus (weeks 11–16) → proves Raft/Paxos
- 3-node cluster via `openraft`: leader election, log replication, snapshots,
  dynamic membership.
- Strong-consistency read/write path for the cache metadata/manifest.
- Jepsen-style fault injection (kill leader, partition, clock skew) → show
  linearizability holds / where it breaks.
- **Deliverable:** chaos-test report; recovery-time-objective numbers.

### Phase 4 — KVCache-for-inference specialization (weeks 17–22) → the role's core
- Prefix/blockwise KV-block cache keyed by token-prefix hash; RAM→SSD tiering;
  admission + eviction (e.g. cost-aware LRU/LFU).
- Integrate with Sophia's inference path (or a vLLM-style mock) to show cache-hit
  speedups on repeated prefixes.
- **Deliverable:** hit-rate vs latency curves; `RESULTS.md` entry.

### Phase 5 — Stretch differentiators (ongoing)
- Zero-copy: `mmap` reads, `splice` on the network path.
- RDMA transport (`async-rdma`) for inter-shard fetch; compare vs TCP.
- FUSE mount exposing the object store; profile the I/O stack with `perf`.
- Write it up to FAST/OSDI workshop standard — *that's* the 同等工程影响力 line.

---

## 4. How this stays true to Sophia's principles

This isn't a bolt-on. It inherits Sophia's culture and that consistency is part
of the story you tell:
- **Fail-closed:** the Rust tier is opt-in; absence falls back to the in-memory
  path. No silent data loss.
- **Honest measurement:** every phase lands real numbers in `RESULTS.md`, no
  inflated claims — same discipline as the rest of the repo.
- **First principles over glue:** the deliverable that matters most is the
  *written tradeoff analysis*, not the lines of code.

---

## 5. Curriculum (papers, in reading order)

1. Raft — *In Search of an Understandable Consensus Algorithm* (Ongaro & Ousterhout)
2. *The Google File System* (SOSP'03) — the canonical DFS shape
3. WiscKey (FAST'16) — key/value separation; LSM amplification
4. *FoundationDB* (SIGMOD'21) — transactions + deterministic simulation testing
5. *Efficient IO with io_uring* (Jens Axboe) — the modern Linux I/O API
6. FaRM (NSDI'14) — RDMA + distributed transactions
7. Mooncake (2024) — KVCache-centric disaggregated inference serving (most on-point)
8. *Facebook's Tectonic Filesystem* (FAST'21) — exabyte-scale object/FS design
9. DeepSeek **3FS** + **Smallpond** — read their own open-source storage stack
   before any interview; align your vocabulary with theirs.

> Reproduce **one** of these (Raft is the highest-leverage). "I implemented Raft
> and broke it under partition, here's the report" beats ten read papers.

---

## 6. First concrete action

If/when you want code: `cargo new --lib storage/kvcache` inside this repo, on the
`claude/distributed-storage-repo-dev` branch, and start Phase 1. The Python side
already has the seam — `agent/vector_store.py` is the integration point.

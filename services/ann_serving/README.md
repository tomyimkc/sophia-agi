# sophia-ann-serving

A dependency-free **Rust** nearest-neighbour core for Sophia's dense recall — the
architecture-track counterpart to the Python retrieval layer (`agent/vector_store.py`,
`agent/hybrid_retrieval.py`). It exists to make the JD's "在成本、延迟与效果之间寻找最优平衡"
(balance cost, latency, and quality) **measurable** rather than asserted.

## What's here

| Type | Search | Use |
|------|--------|-----|
| `FlatIndex` | brute-force exact cosine, O(n·d)/query | recall ground truth; right choice below a few thousand vectors |
| `NswIndex` | navigable small-world graph + greedy beam search | single-layer graph; trades a sliver of recall for a large latency win as *n* grows |
| `HnswIndex` | **multi-layer** HNSW (hierarchical NSW) | adds sparse upper "express-lane" layers over the base graph; **higher recall at the same `ef`** than single-layer NSW |
| `ShardedHnsw` | **N** HNSW shards, id-hash routed, **parallel** fan-out + merge | horizontal scale-out: N× capacity across memory budgets, recall rises via the merge; the architecture-track "千亿级" primitive |

Both `HnswIndex` and `ShardedHnsw` **persist** (`to_bytes`/`from_bytes`) so a built graph is
saved once and reloaded in milliseconds instead of rebuilt — the "build once, serve many" path.

Vectors are assumed L2-normalised, so cosine == dot product — matching the Python
`local-hash-v1` embedder (`agent/rag_local_embed.py`), which already emits unit vectors. The
two indexes share one vector set so the trade-off curve is apples-to-apples.

## Run

```sh
cd services/ann_serving
cargo test                       # flat ordering, NSW≈flat, HNSW≥NSW at equal ef, parser round-trip
cargo run --release --bin bench  # NSW vs HNSW recall/latency trade-off sweep over ef
```

## Measured trade-off (NSW vs HNSW)

`bench` on 20 000 random 64-d unit vectors, 200 queries, k=10, m=16 (deterministic seeded
LCG — reproduces bit-for-bit). Random high-dimensional vectors are the **worst case** for a
graph index; real embeddings cluster and do better. The hierarchy clearly lifts recall at
fixed `ef`. Representative run:

| ef  | NSW recall@10 | HNSW recall@10 | HNSW latency (µs) | speedup vs flat |
|----:|--------------:|---------------:|------------------:|----------------:|
| 16  | 0.32 | 0.52 | ~94  | ~28× |
| 64  | 0.63 | 0.83 | ~290 | ~9× |
| 128 | 0.77 | 0.92 | ~538 | ~5× |
| 256 | 0.87 | **0.96** | ~859 | ~3× |

`ef` (search beam width) is the dial: bigger `ef` buys recall with latency. Flat exact is
~2 600 µs/query here, so HNSW reaches 0.96 recall at ~3× speedup, or trades recall for ~28×.
(Absolute numbers are machine-dependent; the *shape* is the point.)

## Scale-out & persistence

```sh
python tools/export_rag_index.py                        # embeddings.npz → vectors.txt
cargo build --release                                    # serve + pack + bench
# build a 4-shard index ONCE and persist it:
./target/release/pack ../../rag/index/vectors.txt sophia.idx 4 16 200
# serve loads the .idx instantly (no rebuild); or build sharded from text with --shards:
./target/release/serve sophia.idx
./target/release/serve ../../rag/index/vectors.txt --shards 4 --save sophia.idx
```

`ShardedHnsw` splits vectors across shards by a stable id hash (even split, same id → same
shard) and fans a query out to all shards in parallel (`std::thread::scope`), merging per-shard
top-k into a global top-k — **deterministically** (sorted by `(-similarity, id)`, identical to
the sequential merge, verified in tests). Persistence (`pack` / `serve <file>.idx`) skips graph
construction on startup.

**Measured (`bench`, 20k×64-d, ef=128):** sharding *raises* recall — 1→0.92, 4→0.98, 8→0.99 —
because the merge sees more candidates, and adds N× capacity. The parallel **latency** win needs
per-shard work to exceed thread-spawn cost; at this small scale fresh per-query threads cost more
than they save, so production uses larger shards (or a persistent pool — see next steps). Honest:
sharding here is a **capacity + recall** win, not a latency win at toy scale.

## Serving bridge (Python ↔ Rust)

`serve` builds (or loads) the index, then answers queries from stdin — the Rust half of the
bridge. The Python side drives it fail-soft (`AnnClient(shards=N)`, or point it at a `.idx`):

`agent/ann_client.py` spawns `serve`, reads the `READY <n> <dim>` handshake, and issues
`<k> <ef> f0 f1 …` queries — returning row ids that map straight back to the loaded chunks. If
the binary or `vectors.txt` is missing, `available()` is `False` and callers fall back to the
pure-Python vector path. Verified: at high `ef` the bridge returns the **same** top-k as Python
exact cosine over the committed index (`tests/test_ann_bridge.py`).

## Honest bounds & next steps

- **Persistent thread pool.** Search fans out with fresh per-query threads (`thread::scope`) —
  correct and dependency-free, but the spawn cost only pays off when per-shard work is large. A
  reusable worker pool (or distributing shards across processes/machines) is where the parallel
  latency win lands at production scale.
- **In-process PyO3 binding.** The current bridge is a subprocess over a text protocol (robust,
  zero-build-coupling). A `cdylib` + PyO3 binding would remove the process boundary and the
  float-text serialization for lower per-query overhead.
- **Distribution & RDMA.** Shards are in-process; cross-machine sharding with RDMA transport, and
  HNSW deletion/compaction, are the genuinely large architecture items the JD lists as bonus.

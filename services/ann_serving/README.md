# sophia-ann-serving

A dependency-free **Rust** nearest-neighbour core for Sophia's dense recall — the
architecture-track counterpart to the Python retrieval layer (`agent/vector_store.py`,
`agent/hybrid_retrieval.py`). It exists to make the JD's "在成本、延迟与效果之间寻找最优平衡"
(balance cost, latency, and quality) **measurable** rather than asserted.

## What's here

| Type | Search | Use |
|------|--------|-----|
| `FlatIndex` | brute-force exact cosine, O(n·d)/query | recall ground truth; right choice below a few thousand vectors |
| `NswIndex` | navigable small-world graph + greedy beam search | the search/insert core of HNSW (single layer); trades a sliver of recall for a large latency win as *n* grows |

Vectors are assumed L2-normalised, so cosine == dot product — matching the Python
`local-hash-v1` embedder (`agent/rag_local_embed.py`), which already emits unit vectors. The
two indexes share one vector set so the trade-off curve is apples-to-apples.

## Run

```sh
cd services/ann_serving
cargo test                       # correctness: flat ordering, NSW≈flat, recall@10 ≥ 0.85 (n=800)
cargo run --release --bin bench  # recall/latency trade-off sweep over ef
```

## Measured trade-off

`bench` on 20 000 random 64-d unit vectors, 200 queries, k=10, m=16 (deterministic seeded
LCG — reproduces bit-for-bit). Random high-dimensional vectors are the **worst case** for a
single-layer graph; real embeddings cluster and do better. Representative run:

| ef  | latency (µs/query) | recall@10 | speedup vs flat |
|----:|-------------------:|----------:|----------------:|
| 16  | ~44  | 0.33 | ~58× |
| 64  | ~129 | 0.64 | ~20× |
| 128 | ~218 | 0.78 | ~12× |
| 256 | ~408 | 0.87 | ~6.4× |

`ef` (search beam width) is the dial: bigger `ef` buys recall with latency. Flat exact is
~2 600 µs/query here, so even the high-recall NSW setting is several-fold faster, and the
low-recall setting is ~50×. (Absolute numbers are machine-dependent; the *shape* is the point.)

## Honest bounds & next steps

- **Single layer, not full HNSW.** The hierarchical layers that lift recall at fixed `ef` are
  the documented next step; this ships the NSW insert/search primitive they're built on.
- **Not yet wired to Python.** A `cdylib` + PyO3 (or a memory-mapped index file the Python
  `vector_store` can load) would let `agent.hybrid_retrieval` call this for the dense view at
  scale. The fusion layer (`reciprocal_rank_fusion`) is already index-size-agnostic, so only
  the dense backend swaps.
- **In-memory only.** No persistence/sharding/RDMA — those are the genuinely large
  architecture-track items the JD lists as bonus.

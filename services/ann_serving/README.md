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

## Serving bridge (Python ↔ Rust)

`serve` builds an index from an exported vectors file once, then answers queries from stdin —
the Rust half of the bridge. The Python side drives it fail-soft:

```sh
python tools/export_rag_index.py                       # rag/index/embeddings.npz → vectors.txt
cargo build --release --bin serve                       # build the server
python -c "from agent.ann_client import AnnClient; \
  import agent.rag_local_embed as e, agent.vector_store as v; \
  idx=v.load_index(v.index_dir()); \
  c=AnnClient(); \
  print(c.search.__doc__) if not c.available() else None"
```

`agent/ann_client.py` spawns `serve`, reads the `READY <n> <dim>` handshake, and issues
`<k> <ef> f0 f1 …` queries — returning row ids that map straight back to the loaded chunks. If
the binary or `vectors.txt` is missing, `available()` is `False` and callers fall back to the
pure-Python vector path. Verified: at high `ef` the bridge returns the **same** top-k as Python
exact cosine over the committed index (`tests/test_ann_bridge.py`).

## Honest bounds & next steps

- **In-process PyO3 binding.** The current bridge is a subprocess over a text protocol (robust,
  zero-build-coupling). A `cdylib` + PyO3 would remove the process boundary and the float-text
  serialization for lower per-query overhead — a pure optimization over the working bridge.
- **Text vectors format.** `serve` reads `id f0 f1 …` text; a memory-mapped binary index
  (`f32` blocks) would cut load time and memory at scale.
- **In-memory only.** No persistence/sharding/RDMA, and HNSW deletion is not implemented —
  those are the genuinely large architecture-track items the JD lists as bonus.

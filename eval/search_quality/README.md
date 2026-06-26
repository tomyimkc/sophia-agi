# Search-quality benchmark (搜索质量评估体系)

An automated, **explainable** search-quality harness — the JD's *搜索质量评估体系：自动化评估、
badcase 分析* — over Sophia's committed RAG index. It turns "which retrieval backend is better"
into measured, graded numbers plus a badcase taxonomy that attributes every miss to a stage.

Harness: [`tools/eval_search_quality.py`](../../tools/eval_search_quality.py).
Guard test: [`tests/test_eval_search_quality.py`](../../tests/test_eval_search_quality.py).

## What it measures

Three backends over the **same** index, so the deltas are causal:

- **keyword** — lexical token-overlap (`agent.retrieval`, `SOPHIA_RAG_BACKEND=keyword`);
- **vector** — dense cosine over `local-hash-v1` embeddings;
- **hybrid** — dense+sparse weighted Reciprocal Rank Fusion (`agent.hybrid_retrieval`).

Graded metrics (exact record = gain 3, any chunk *about* it = gain 1):

- **recall@k** and **MRR** keyed on the exact canonical record;
- **nDCG@k** — quality, not just hit/miss. IDCG is **pooled** (the ideal is the best gain
  ordering any backend surfaced for that probe), an honest small-corpus approximation.

## Badcase taxonomy

Every probe whose exact record the backends mishandle is bucketed so iteration has a clear,
explainable target:

| Bucket | Meaning | Actionable signal |
|--------|---------|-------------------|
| `lexical_gap` | keyword misses the exact record but a vector view finds it | dense recall is pulling its weight; don't ship keyword-only |
| `semantic_gap` | vector misses but keyword/hybrid finds it | the hash embedding blurred a rare/exact term; sparse must stay in the fusion |
| `tied_burial` | exact is retrievable (in the pool) but buried below top-k | a ranking/tie-break problem, not a recall hole — rerank target |
| `absent_from_pool` | no backend surfaces the exact record | a true recall hole — indexing/chunking/embedding target |

## Run

```sh
python tools/eval_search_quality.py            # full run → writes the candidate report
python tools/eval_search_quality.py --json      # machine-readable summary
python tools/eval_search_quality.py --limit 8   # quick smoke
```

Report: `agi-proof/benchmark-results/search-quality.public-report.json`.

## What the current run shows (honest)

On this corpus the **dense vector** backend wins outright (highest recall@5 and nDCG@5);
**hybrid** beats keyword and closes real `lexical_gap` badcases but does **not** beat pure
dense — because the corpus is dominated by near-duplicate teacher examples on which the sparse
BM25 view is high-recall / low-precision. That finding is the point: the harness *revealed*
that sparse needs near-duplicate dedup (and weighting) before fusion pays off here, rather
than assuming hybrid is universally better. Weighted RRF (dense 1.0 / sparse 0.4) is the
current mitigation; per-corpus tuning is expected.

## Honest bounds

Probes are **self-authored** over the live corpus (not a third-party retrieval benchmark);
scoring is exact-match-against-gold (no LLM judge); nDCG uses pooled IDCG. So this validates
the **ranking deltas and the harness end-to-end** — a reproducible, offline quality signal —
not a headline capability number. `candidateOnly: true`, `validated: false` in the report.

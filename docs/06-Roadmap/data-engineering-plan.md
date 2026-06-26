# Pretraining Data-Engineering Plan

**Status:** active · **Branch:** `claude/pretraining-data-engineering-8l1z3u`

A phased plan to extend Sophia's existing provenance/trust layer into a credible
**pretraining-data-engineering** pipeline (web-scale acquisition → cleaning →
dedup → quality scoring → columnar sharding → catalog), without contradicting
[VISION.md](../../VISION.md). Each phase ships a standalone, testable artifact and
**reuses** existing `agent/` primitives rather than bolting on a generic crawler.

## North star

> Sophia already owns the hard, differentiated half of a data pipeline —
> provenance-aware **quality/trust scoring** (`agent/poison_resistant_ingestion.py`,
> `agent/grounded_confidence.py`, `agent/gate.py`) and **contamination/leakage
> guards** (`provenance_bench/dataset_guard.py`). This plan adds the missing half —
> **acquisition, dedup, columnar processing, infra** — and wires the two together so
> data *selection* is driven by quality signals.

New top-level package: **`pipeline/`** (collection → clean → dedup → score → shard →
catalog), importable from `agent/` so it stays consistent with the trust-layer vision.

## Guardrails (preserve repo identity)

- Every stage emits provenance + quality metadata — no anonymous data.
- Dedup and quality regression are **fail-closed** (mirror `dataset_guard`).
- Airgap/local-first default kept: deterministic embedder, no mandatory external service.

## Phases

| # | Phase | Ships | JD direction | Status |
|---|-------|-------|--------------|--------|
| 0 | Foundation & contracts | `pipeline/` pkg, document schema, manifest, fixtures | all | ✅ done |
| 1 | Quality scoring | `quality_score.py`, `link_priority.py`, `tools/score_corpus.py` | 采集 pipeline (链接质量预估/数据优先级) | ✅ done |
| 2 | Dedup & URL hygiene | MinHash-LSH, vector near-dup, URL canonicalization | 采集 / 语言数据 (MinHash/向量去重/链接规模控制) | ✅ done |
| 3 | Columnar & regression | DuckDB/Parquet corpus stats, quality-regression gate | 语言数据处理 (质量回归) | ✅ done |
| 4 | Acquisition loop | async crawler, priority frontier, CommonCrawl WARC ingest | 采集 pipeline (全网采集环路) | ✅ done |
| 5 | Scale proof & infra | object-store shards, KV seen-set, queue, RESULTS numbers | 数据基建 (TB tokens) | ⏳ |
| + | Multimodal slice (stretch) | image-text WebDataset shards w/ perceptual-hash dedup | 多模态数据 | ⏳ |

### Phase 0 — Foundation & contracts
- `pipeline/schemas/document.schema.json` — canonical document record.
- `pipeline/document.py` — `Document` validation (stdlib, no jsonschema dep).
- `pipeline/manifest.py` — shard catalog (path, rows, dedup ratio, quality histogram, sha256).
- `tests/fixtures/pipeline_docs.jsonl` — tiny offline corpus (incl. near-dup, mirror, low-trust).
- **Acceptance:** schema validates fixtures; manifest round-trips; CI green.

### Phase 1 — Quality scoring (链接质量预估 / 数据优先级)
- `pipeline/quality_score.py` — `score_document(doc)` composing: source trust + k≥2
  corroboration (`assess_item`), `authorConfidence` priors (`AUTHOR_CONFIDENCE_PRIOR`),
  and cheap heuristics (length, alpha/symbol ratio, boilerplate, language purity).
- `pipeline/link_priority.py` — aggregate per registered-domain → crawl priority + quota.
- `tools/score_corpus.py` — CLI: score a JSONL batch, emit a quality histogram + priority report.
- **Acceptance:** low-trust/boilerplate docs score below curated docs; deterministic; tested.

### Phase 2 — Dedup & URL hygiene (MinHash / 向量去重 / 链接规模控制) ✅
- `pipeline/url_canonical.py` — strip tracking/session params, normalize host/scheme/port/slash,
  collapse URL variants (`canonical_clusters`).
- `pipeline/dedup/minhash.py` — MinHash + LSH near-dup, **pure stdlib** (no datasketch);
  stable `blake2b` hashing, deterministic; estimate tracks true Jaccard within ~0.003 on test.
- `pipeline/dedup/vector.py` — embedding near-dup via `agent/rag_local_embed.py` (airgap-safe;
  numpy optional, `available()` guard + MinHash fallback).
- `pipeline/dedup/__init__.py` — `dedup_documents` stage: canonicalize → MinHash cluster →
  set `doc['dedup']`, keep one per cluster, report dedup ratio. Wired into `tools/score_corpus.py --dedup`.
- **Done:** mirror pair in fixtures detected & removed; param-variant URLs collapse; tests green.

### Phase 3 — Columnar & quality regression (duckdb / 质量回归) ✅
- `pipeline/io.py` — shard read/write; **Parquet when pyarrow present, JSONL fallback** (lazy
  import, so it runs in a minimal/airgapped env). JSONL stays the deterministic interchange.
- `pipeline/corpus_table.py` — analytical summary (count, tokens, lang/quality histograms,
  domain counts, keep/duplicate rates). Uses **DuckDB over Parquet when available**, else a
  stdlib aggregation returning the identical shape; CI exercises the stdlib path.
- `pipeline/quality_regression.py` — **fail-closed** gate: flags mean-quality drop, keep-rate
  drop, duplicate-rate spike, or token-volume collapse vs a committed baseline (mirrors
  `dataset_guard` discipline).
- `tools/pipeline_stats.py` — CLI: summarize a shard, write a baseline summary, and gate a
  build against it (non-zero exit on regression).
- *Note:* kept `tools/corpus_stats.py` (README badges) intact; added a new pipeline-scoped
  stats tool rather than break it. Heavy engines (`duckdb`/`pyarrow`) are **optional** — see
  `requirements-pipeline.txt`; CI currently runs the stdlib fallback path.
- **Done:** end-to-end score → dedup → shard → stats → gate verified; regression fails closed.

### Phase 4 — Acquisition loop (全网采集环路) ✅
- `pipeline/fetch/frontier.py` — max-priority URL queue, canonical-dedup, `feedback()` re-rank.
- `pipeline/fetch/robots.py` — async robots.txt gating (`urllib.robotparser`), conservative on error.
- `pipeline/fetch/crawler.py` — **async** crawl over an **injectable transport** (mock/airgap;
  httpx in prod): per-host rate limit + fetch quota + global `max_pages` (抓取配额), retry with
  exponential backoff on 5xx/429/transport error. Injectable clock/sleep → deterministic tests.
- `pipeline/fetch/extract.py` — dependency-free `<a href>` link + text extraction.
- `pipeline/fetch/warc.py` — WARC `response`-record ingest (gzip-aware) → documents; the
  network-free path to run at scale against an archived crawl.
- `pipeline/fetch/loop.py` — `run_loop`: seed → fetch → extract → **score (Phase 1) → re-rank
  frontier from that quality** → repeat. This is the closed selection loop, not a blind spider.
- `tools/crawl_warc.py` — WARC → extract → dedup → score → shard + manifest.
- **Done:** loop follows discovered links and scores; WARC CLI removes a mirror (33% dedup on
  the sample) and emits a scored shard; 12 fetch tests green (all offline via mock transport).

### Phase 5 — Scale proof & infra (数据基建)
- Run 1–4 over a multi-GB public dump (CommonCrawl/Wikipedia/OSCAR).
- Parquet shards in object store (MinIO/S3) + manifest catalog; KV (RocksDB/Redis) seen-set +
  dedup fingerprints; Redis Streams/NATS replacing the single-process JSONL queue.
- Publish a RESULTS-style table: docs processed, GB in/out, dedup ratio, pass rate, throughput, cost.

### Stretch — Multimodal slice
- One image-text pipeline → perceptual-hash dedup → quality filter → WebDataset/Parquet shards,
  each sample carrying `okf/` provenance lineage.

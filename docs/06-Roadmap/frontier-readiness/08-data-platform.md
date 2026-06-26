# 08 — Training-Data Platform: from a polite crawler to a FineWeb/Dolma-grade pipeline

**Repo:** `/home/user/sophia-agi`
**Author role:** Staff data-platform / data-infrastructure engineer
**Status:** implementation plan + research thesis
**Date:** 2026-06-26
**Posture (non-negotiable):** This plan scales a *real* data pipeline to credible large-scale data-platform experience (Anthropic Research Data Platform / RL Data / Data Engineer; OpenAI Data Platform — Spark/Iceberg/Kafka/Flink). It does **not** claim petabyte scale today. Every milestone produces a **reproducible, measured artifact** (content hash + manifest + datasheet), and the headline differentiator — a per-row **data passport / provenance / lineage** layer with a fail-closed contamination guard — is preserved and pushed to the center. No AGI claim; `candidateOnly: true`; metrics are measured, not asserted.

---

## 1. Thesis & references

### 1.1 The claim
The frontier-lab data role is not "write a crawler." It is to **own the data supply chain end-to-end**: acquire web-scale raw bytes, turn them into a *cleaned, deduplicated, decontaminated, quality-filtered, versioned, lineage-tracked* training corpus, and prove — reproducibly, with numbers — that the corpus is what you say it is. The hard parts are (a) **scale** (process CommonCrawl-sized shard sets on distributed compute), (b) **correctness at scale** (dedup that actually deduplicates billions of docs; quality filters whose effect on downstream loss is *measured*), and (c) **trust** (provenance, license tracking, test-set decontamination, dataset versioning so an experiment is reproducible from a catalog snapshot). A credible candidate has shipped a pipeline that does all three on real CommonCrawl shards and can show the dedup rate, the filter survival curve, the decontamination coverage, and a versioned catalog that a training job reads.

The published methodologies define the bar, and this plan implements their *shape* honestly at a tier the available compute supports:

- **RefinedWeb** (Penedo et al., 2023) — web-only data can match curated corpora *if* you process it hard: trafilatura-grade text extraction from WARC, aggressive **MinHash near-dedup** + exact dedup, URL/line-level filtering, and Gopher-style quality heuristics. The headline lesson: dedup and filtering are where the quality comes from, not the source.
- **FineWeb / FineWeb-Edu** (Penedo et al., HuggingFace, 2024) — a fully reproducible 15T-token CommonCrawl pipeline (the `datatrove` library): per-dump processing, **MinHash LSH dedup applied *within* each dump** (a deliberate, measured choice — global dedup over-removed and *hurt* downstream eval), C4/Gopher quality filters, and a **model-based educational-quality classifier** (FineWeb-Edu). The discipline to emulate: every filter decision is justified by an **ablation on downstream benchmark loss**, not by intuition.
- **Dolma** (Soldaini et al., AI2, 2024) — a 3T-token open corpus *and* the `dolma` toolkit: a **taggers → mixer → dedup (Bloom-filter)** architecture, explicit **PII removal**, license/source tracking, and a published **datasheet**. The discipline to emulate: data documentation as a first-class artifact.
- **The Pile / Gopher / C4 / MassiveText** — quality-filter heuristics (stop-word ratio, mean line length, symbol-to-word ratio, fraction of lines ending in ellipsis, duplicate-line fraction), and the original case for exact + fuzzy dedup.
- **MinHash + LSH dedup** (Broder, 1997; "Deduplicating Training Data Makes Language Models Better", Lee et al., 2021) — the canonical fuzzy near-dup method: shingle → MinHash signature → LSH banding → union-find clusters; plus suffix-array exact-substring dedup as the complementary exact pass.
- **Decontamination** — n-gram / shingle containment of eval items against training text (used by GPT-3, Llama, FineWeb, OLMo). The honest framing: catches verbatim/near-verbatim leakage, **not** semantic paraphrase — and you report coverage, not a clean bill of health.
- **Data lake / table formats** — **Apache Iceberg** (hidden partitioning, snapshot isolation, time-travel, schema evolution) and **Delta Lake** (ACID over Parquet, transaction log). The platform value: a corpus version *is* a table snapshot; an experiment pins a snapshot id and is reproducible forever.
- **Distributed compute** — **Apache Spark** (the industry default for PB-scale ETL/dedup; FineWeb's `datatrove` and Dolma both run Spark/Slurm-scale jobs) and **Ray Data** (Python-native streaming datasets, good for GPU-in-the-loop model-based filtering / embedding dedup).
- **Streaming** — **Apache Kafka** (durable ingest log / backpressure for a continuous crawl→process loop) and **Apache Flink** (stateful streaming dedup and windowed quality stats). Relevant for the "always-on acquisition loop" framing, not for batch CommonCrawl.
- **Sharded dataset formats** — **WebDataset** (tar shards, sequential streaming, the multimodal standard) and **MosaicML Streaming / MDS** (random-access shards, deterministic resumption, cloud-streaming for training). The training-loader contract.
- **Dataset versioning / lineage** — LakeFS / DVC / Iceberg snapshots for data; **Croissant** and **Datasheets for Datasets** (Gebru et al., 2018) for documentation. This is where the repo already has a genuine edge.

### 1.2 Why this repo can credibly claim it
Most "I built a data pipeline" portfolios stop at crawl + clean. This repo already has the parts most candidates *don't*: a **provenance-aware quality scorer** that composes a trust layer, a **per-row data passport** with content hash + MinHash + dedup cluster + license flagging + a published **datasheet**, a **fail-closed contamination guard** wired into the dataset build, and a **quality-regression CI gate**. The thesis is therefore not "catch up to FineWeb." It is: **implement the FineWeb/Dolma processing core honestly at the scale our compute supports, and fuse it with a provenance/passport/lineage layer that frontier corpora gesture at (datasheets) but rarely operationalize per-row.** That fusion is the differentiator.

---

## 2. Current repo state (file-level, honest)

A modest but genuinely well-built crawler + WARC + clean/dedup/score pipeline with stdlib-first, deterministic, airgap-safe discipline. It is **not** a large-scale platform: no distributed compute, no table format, dedup is single-process in-memory, "scale" is asserted in docstrings not demonstrated, and there is no CommonCrawl-shard processing harness. Honest inventory:

**Acquisition loop (`pipeline/fetch/`)** — real and clean.
- `frontier.py` — `Frontier`: heapq max-priority URL queue, canonical-form dedup, quality-feedback re-ranking. Deterministic. Single-process, in-memory.
- `loop.py` — `run_loop(seeds, transport)`: seed → fetch → extract text+links → `score_document` → feed outlink priority back from page quality. Transport-injected (mock or httpx). The "selection, not blind spider" loop.
- `crawler.py`, `robots.py`, `http.py`, `extract.py` — async polite crawler (robots, per-host quota, backoff, injectable clock), regex link/text extraction (no lxml/bs4 — airgap-safe but lower-fidelity than trafilatura).
- `warc.py` — dependency-free WARC reader: `iter_warc_records` / `read_warc` (gzip-aware) / `records_to_documents` (keeps `response` text/html). Correct for the subset that matters; **not** a full WARC 1.1 impl, and it reads the **whole file into memory** (`path.read_bytes()` / `gzip ... .read()`) — fine for sample shards, will not stream a real 1 GB CC WARC.

**Clean / dedup / score / catalog (`pipeline/`)** — the core, all single-process.
- `quality_score.py` — **the differentiator.** Blends content heuristics (length, alpha-ratio, boilerplate markers, script purity, repetition) with a **provenance signal** reusing `agent.poison_resistant_ingestion.assess_item` (k-independent, trust-shrunk corroboration) and `agent.grounded_confidence.AUTHOR_CONFIDENCE_PRIOR` (OKF tier prior). Fail-closed: no-provenance capped at 0.7; quarantine penalized. Pure, deterministic. Heuristics are a *subset* of Gopher/C4/RefinedWeb (no stop-word ratio, no line-level filters, no symbol-to-word, no perplexity/model-based classifier).
- `dedup/minhash.py` — from-scratch MinHash + LSH banding + union-find, stable blake2b hashing, deterministic. Correct and well-commented. **In-memory, single-process, O(all docs in RAM)** — not a distributed/Bloom-filter dedup.
- `dedup/vector.py` — embedding near-dup via `agent.rag_local_embed` (hashing embedder) + greedy single-linkage cosine. Catches paraphrase-ish dups MinHash misses. Greedy O(n·reps), in-memory.
- `document.py` — document contract + JSONL read/write; `validate` is fail-closed and returns problems rather than raising. Schema at `schemas/document.schema.json`.
- `manifest.py` — per-shard manifest: rowCount, dedupRatio, quality histogram, **order-independent `contentSha256`**, plus `verify_manifest` for CI drift checks. This is the reproducibility anchor.
- `shard_writer.py` — `write_sharded`: batches docs into `part-NNNNN.jsonl` shards, stamps each with a manifest, writes a `_catalog.json`. This is a hand-rolled mini-catalog — the seed of a table format, but **not** Iceberg/Delta (no snapshots, no schema evolution, no ACID).
- `io.py` — shard I/O: JSONL always, **Parquet when pyarrow present** (else JSONL fallback). Flattens nested blocks to columns.
- `corpus_table.py` — columnar stats (token totals, lang/quality/domain histograms, keep/dup rate); **DuckDB over Parquet when available**, identical stdlib fallback otherwise.
- `quality_regression.py` — fail-closed CI gate: blocks a corpus build if mean-quality drops, keep-rate drops, dup-rate spikes, or token volume collapses past tolerances. Genuinely good ops discipline.
- `link_priority.py`, `url_canonical.py`, `quality_score` feedback — crawl prioritization.

**Storage (`pipeline/store/`)** — backend-agnostic, production-shaped.
- `objectstore.py` — `ObjectStore` Protocol (put/get/exists/list + shard conveniences); `LocalObjectStore`.
- `s3.py` — `S3ObjectStore` over an injected boto3-style client (AWS S3 / MinIO / Ceph RGW), unit-tested with an in-memory fake; `from_env` for real boto3. Solid.
- `kv.py`, `queue.py`, `redis_backends.py` — KV/queue abstractions (Redis-backed optional).

**Multimodal (`pipeline/multimodal/`)** — a real WebDataset path.
- `shards.py` — `write_webdataset` / `read_webdataset`: **real tar shards** (`<key>.jpg/.txt/.json`) via stdlib `tarfile`, loadable by any WebDataset reader; metadata carries provenance + dedup + quality.
- `phash.py` — aHash/dHash perceptual hashing + hamming, stdlib core (PIL optional for decode).
- `process.py`, `sample.py` — sampling/processing.

**Provenance / honesty layer (the edge) — outside `pipeline/` but it's the moat.**
- `pretraining/data_passport/passport.py` — `stamp_pack(rows)`: per-row `_passport` (content_hash, source, license, quality_score, 16-perm MinHash, dedup_cluster, exact_duplicate, flags) + a **`datasheet`** (rows, unique_clusters, duplicate_rate, by_source, by_license, mean_quality, flagged_rows). Fail-closed: unknown license → `unlicensed`, low quality → `low_quality`. This is a *Datasheets-for-Datasets* + Dolma-datasheet idea, per-row.
- `eval/contamination.py` — n-gram shingle **containment** decontamination (eval-item-vs-train); honest that it catches verbatim/near-verbatim, not paraphrase.
- `provenance_bench/dataset_guard.py` — `check_contamination()`: fail-closed prompt-disjointness guard wired into `tools/build_local_sophia_dataset.py`; records `droppedForDecontamination` in the manifest.
- `agent/poison_resistant_ingestion.py`, `agent/grounded_confidence.py`, `agent/corroboration.py` — the trust model `quality_score` composes (k-independent corroboration, source-trust shrinkage, OKF authorConfidence priors, log-odds pooling).
- `okf/` — provenance-native knowledge graph (authorConfidence tiers, tradition-based independence groups).

**Data (`data/`)** — confirmed: this is **eval/benchmark/labeled JSON** (behavioral batteries, council tasks, held-out manifests), **not** a training corpus. Training rows live under `training/` and are built by `tools/build_local_sophia_dataset.py`.

**Tests** — `tests/test_pipeline_*.py` cover fetch, dedup, quality, manifest, shard writer, corpus table, store, cloud adapters, url canonical, multimodal. Good coverage of the *unit* behavior; nothing exercises a real CommonCrawl shard or a distributed run.

**`requirements-pipeline.txt`** — `pyarrow>=14`, `duckdb>=0.10`, `numpy>=1.24`, all optional. No Spark, Ray, Iceberg, Delta, datatrove, datasketch, trafilatura, fasttext.

**Net honest read:** a high-quality *single-node reference implementation* of the FineWeb/Dolma *stages* (extract → clean → MinHash dedup → quality score → shard → catalog → manifest), plus a provenance/passport/contamination layer most pipelines lack. What's missing for the frontier bar: (1) running it on **real CommonCrawl shards** at meaningful volume, (2) **distributed** dedup/filtering (Spark or Ray Data), (3) a **real table format** (Iceberg/Delta) instead of `_catalog.json`, (4) **streaming WARC** + better extraction (trafilatura), (5) the **quality-filter ablation** discipline (measure each filter's effect on a downstream proxy), (6) **PII removal**, (7) **MDS** training-shard output alongside WebDataset.

---

## 3. Top-tier target end-state

A `pipeline/` (+ new `pipeline/cc/`, `pipeline/lake/`, `pipeline/distributed/`) that, for a set of CommonCrawl shards, produces a **versioned, lineage-tracked, decontaminated, deduplicated, quality-filtered training corpus** with the same epistemic discipline the repo already demands of its fact-checker:

1. **WARC → text** at scale: streaming WARC/WET ingest, trafilatura-grade extraction (optional dep, regex fallback preserved), language ID (fastText/CLD3 optional), URL + line-level filters.
2. **FineWeb/Dolma quality filtering**: full Gopher/C4/RefinedWeb heuristic suite, each filter emitting a **tag** (Dolma taggers model), plus an optional **model-based edu/quality classifier** (FineWeb-Edu style) run on Ray Data. Every filter's keep/drop and its **downstream-loss ablation** are recorded.
3. **Dedup that scales**: exact (content-hash / Bloom filter, Dolma-style) + **MinHash LSH** near-dedup, **applied per-dump** (FineWeb's measured choice), runnable single-node (current impl) *and* distributed (Spark partition-banded LSH / Ray). Dedup rate reported and reproducible.
4. **Decontamination as a gate**: every training shard checked (shingle containment) against the repo's held-out eval banks (`data/*_benchmark/heldout_v1.jsonl`, `reference_holdout_traps.json`); **coverage** reported, overlap dropped, recorded in the manifest. Fail-closed.
5. **PII removal**: regex + (optional) NER pass (email/phone/IP/secrets), counts recorded per shard (Dolma discipline).
6. **A real table format**: corpus stored as **Iceberg** (preferred; Delta as alt) over Parquet, partitioned by dump/lang/quality-bucket. A corpus version = a **snapshot id**; training pins it (time-travel reproducibility).
7. **Training-ready shards**: **WebDataset** (have) + **Mosaic MDS** (new) emitted from the catalog, deterministic-resumable.
8. **Data passport / lineage layer (the differentiator)**: every output row keeps its **passport** (content hash, source, license, dedup cluster, every filter tag it passed/failed, the snapshot it landed in); a **lineage record** ties shard → dump → WARC → URL and filter-version. A published **datasheet** per corpus version. This is the artifact a frontier data team would actually trust.
9. **Compute tiers**: identical logic runs on (a) local sample shards (CI, stdlib), (b) a single big box / DGX Spark lane (the repo already has a `spark-gpu` CI lane), (c) **RunPod / cloud Spark or Ray** for a real multi-shard run.
10. **Ops**: the `quality_regression.py` gate guards every build; manifests + snapshot ids make every number reproducible from a hash.

---

## 4. Phased plan — milestones

Guiding rule: **each milestone ends in a reproducible artifact** (manifest contentSha256 + datasheet + Iceberg snapshot id) and a **measured number**, never a docstring claim. Start where the leverage is highest: a real FineWeb/Dolma-style processing run on actual CommonCrawl sample shards, output as WebDataset/MDS + a table catalog. Then scale the compute. Then make the passport/lineage layer the headline.

### M1 — Real FineWeb/Dolma processing on CommonCrawl sample shards (single-node)
*The credibility unlock: stop asserting "TB-scale", actually process real CC shards end-to-end.*
- **New `pipeline/cc/`**:
  - `cc/fetch_sample.py` — download a handful of real CommonCrawl WARC/WET segments by `crawl-data/CC-MAIN-*/segments/.../warc.gz` path (a few GB total; index via the CC `cc-index` or a pinned segment list). Uses `S3ObjectStore`/HTTP; records source URLs.
  - `cc/warc_stream.py` — **streaming** WARC/WET reader (refactor `pipeline/fetch/warc.py` to iterate over a file object, not `read_bytes()`), so a 1 GB shard processes in bounded memory.
  - `cc/extract.py` — trafilatura extraction (optional dep) with the existing regex extractor as fallback; language ID (fastText `lid.176` optional, heuristic fallback).
- **Quality filters → Dolma taggers model**: extend `pipeline/quality_score.py` into `pipeline/filters/` — add the full Gopher/C4/RefinedWeb heuristics (stop-word ratio, mean line length, symbol-to-word ratio, fraction of lines ending in ellipsis/bullet, duplicate-line fraction, repeated-n-gram fraction). Each filter is a **tagger** that emits a named tag + value; the keep decision is the composition. Preserve the provenance signal as one tagger.
- **Decontamination wired into the corpus build**: a `pipeline/decontam.py` that runs `eval/contamination.py` containment of each shard against the repo's held-out banks; drop overlapping docs, record `droppedForDecontamination` + **coverage** (fraction of eval shingles checked) in the manifest. Reuse `provenance_bench/dataset_guard.py` patterns; fail-closed.
- **PII pass**: `pipeline/pii.py` — regex email/phone/IP/IBAN/secret-key + optional NER; redact + count per shard.
- **Output**: reuse `shard_writer.py` → JSONL/Parquet shards + `_catalog.json` + per-shard manifest. Add **MDS** writer `pipeline/mds.py` (Mosaic Streaming format) alongside the existing WebDataset writer.
- **CLI**: `python -m pipeline.cc.run --segments segments.txt --out corpus/cc-sample-v1` producing a corpus + datasheet.
- **Artifact:** a real (small, e.g. 1–5 GB raw → N00k–Mk docs) CommonCrawl-derived corpus with a manifest (contentSha256), a datasheet (dedup rate, keep rate, per-filter drop counts, PII counts, decontam coverage), as JSONL/Parquet + WebDataset + MDS.
- **Tools:** stdlib + pyarrow/duckdb; trafilatura, fastText, mosaicml-streaming as optional deps. **Compute:** local / single box.

### M2 — A real table format + dataset versioning (Iceberg/Delta catalog + lineage)
*Replace the hand-rolled `_catalog.json` with a snapshot-versioned lake; make a corpus version reproducible from a snapshot id.*
- **New `pipeline/lake/`**:
  - `lake/iceberg_catalog.py` — write the corpus as an **Apache Iceberg** table (PyIceberg + a local/S3 warehouse + a SQLite/REST catalog), partitioned by `dump`, `lang`, `quality_bucket`. (Provide a **Delta Lake** alt via `delta-rs`/`deltalake` behind the same interface; keep the `_catalog.json` path as the airgap fallback.)
  - `lake/snapshot.py` — pin/resolve a corpus version to a **snapshot id**; `time_travel(snapshot_id)` returns the exact shard set a training run saw.
  - `lake/lineage.py` — **the lineage layer**: a record per shard tying `shard → iceberg_snapshot → dump → warc_path → source_urls`, plus the **filter-version** and **passport** of each row. Lineage is queryable (DuckDB/Iceberg) and committed as JSON for airgap.
- **Wire the passport in**: `pretraining/data_passport/passport.py` becomes the per-row lineage carrier — every Iceberg row keeps `_passport` (content hash, source, license, dedup cluster, filter tags). The Dolma-style **datasheet** is generated *from* the Iceberg snapshot.
- **Reproducibility test**: CI asserts that resolving a pinned snapshot id reproduces the exact `contentSha256` set (extends `manifest.verify_manifest` to the table level).
- **Artifact:** the M1 corpus, now an Iceberg table with a named snapshot, a lineage graph, and a datasheet generated from the snapshot. A training config can pin `snapshot=...` and be reproducible.
- **Tools:** PyIceberg (or deltalake), pyarrow, duckdb. **Compute:** local / single box.

### M3 — Scale the compute: distributed dedup + filtering (Spark or Ray Data)
*Take the single-node logic to a real multi-shard run. This is the "at scale" proof.*
- **New `pipeline/distributed/`** — port the *same* stages to a distributed engine behind a thin interface so local and distributed paths share filter/dedup code:
  - **Option A — Apache Spark** (matches the OpenAI/Anthropic JD "Spark at scale" and the repo's existing `spark-gpu` CI lane): `distributed/spark_dedup.py` implements **partition-banded MinHash LSH** (shingle→signature in a UDF, band→bucket repartition, union-find per bucket, global cluster reduce). `distributed/spark_filter.py` runs the taggers as Spark transforms. Reads CC WARC/WET via `binaryFiles`/S3, writes Iceberg (Spark has native Iceberg + Delta support).
  - **Option B — Ray Data** (Python-native, best for the **model-based** edu/quality classifier and embedding dedup with GPUs): `distributed/ray_pipeline.py` streams shards, runs taggers + an optional fastText/transformer quality classifier on a Ray actor pool, and an embedding near-dedup on GPU.
  - Pick **Spark** as the primary (table-format integration + the JD bar); keep **Ray Data** for the GPU-in-the-loop classifier stage.
- **Validation:** run on a **larger** CC slice (tens of GB → tens of M docs) on RunPod / cloud. Show the distributed dedup rate **matches** the single-node rate on an overlapping sample (correctness), and report throughput (docs/hour, $/Mdoc).
- **Artifact:** a multi-shard corpus processed distributed, Iceberg-cataloged, with a throughput + cost report and a single-node-vs-distributed dedup-rate agreement check.
- **Tools:** PySpark + Iceberg/Delta; Ray Data; fastText/transformers (classifier). **Compute:** RunPod multi-node or a single large box for Spark local-cluster; cloud Spark/EMR-style as the stretch.

### M4 — The differentiator: data-passport / provenance / lineage as a first-class product
*Make the trust layer the headline — this is what other portfolios don't have.*
- **Passport everywhere**: every row in the Iceberg corpus carries a full passport (content hash, source URL, license, dedup cluster id, the ordered list of **filter tags it passed/failed**, the **decontam verdict**, the **PII redaction count**, and the **snapshot** it belongs to). A row's history is auditable end-to-end.
- **Lineage queries**: "show me every training row derived from domain X / dump Y / that failed filter Z / that was dropped for contamination" — answerable from the Iceberg/lineage tables.
- **Datasheet generator** (`pipeline/datasheet.py`): produces a Datasheets-for-Datasets + Dolma-style **published datasheet** per corpus version (motivation, composition, collection, preprocessing/dedup/decontam/PII, licenses, known limitations). Generated from the snapshot, committed, diffable.
- **Contamination report as a standing artifact**: decontam coverage + per-eval-bank overlap, versioned per corpus snapshot, gated by `quality_regression.py`.
- **Honest-metrics dashboard**: the `corpus_table.py` summary extended to a per-snapshot report (docs, tokens, dedup rate, filter survival curve, decontam coverage, PII counts, license breakdown) checked into the repo per build.
- **Artifact:** a **trusted corpus**: Iceberg snapshot + lineage graph + passport per row + published datasheet + decontamination report. This is the single highest-signal deliverable.
- **Tools:** existing passport/contamination code, Iceberg, DuckDB. **Compute:** local (runs over the M3 catalog).

### M5 (stretch) — Streaming acquisition loop (Kafka/Flink) + model-based filtering at scale
*Only if the batch story is solid. Demonstrates the streaming half of the JD.*
- `pipeline/stream/` — a Kafka ingest log for the continuous crawl→process loop (the existing `Frontier`/`run_loop` produces to Kafka; processors consume), with **Flink** stateful dedup (keep a rolling MinHash/Bloom state) + windowed quality stats. Backpressure-correct.
- FineWeb-Edu-style **model-based quality classifier** trained on the repo's quality scores as weak labels, run on Ray Data at scale, with the **downstream-loss ablation** proving it helps.
- **Artifact:** a streaming pipeline processing a live crawl shard-by-shard with stateful dedup, and a quality-classifier ablation.

---

## 5. Compute / budget tiers

| Tier | Hardware | What runs | Volume | Cost | Purpose |
|---|---|---|---|---|---|
| **T0 — CI / airgap** | laptop, 1 core, stdlib only | unit tests; a tiny synthetic WARC; full pipeline on a few MB | KB–MB | $0 | determinism + regression gate; proves the logic without any cloud |
| **T1 — Local sample** | single box, 8–32 GB RAM, pyarrow/duckdb | M1 end-to-end on **real CC sample shards**; Iceberg local warehouse (M2); passport/datasheet (M4) | 1–5 GB raw → ~100k–1M docs | ~$0 (local) or a few $ egress | the credibility unlock; everything reproducible locally |
| **T2 — DGX Spark / big box** | the repo's `spark-gpu` lane / one large instance, Spark local-cluster | M3 distributed dedup/filter on a bigger slice; Ray Data classifier on 1 GPU | tens of GB → tens of M docs | ~$5–40 for a RunPod GPU/CPU box for a few hours | proves the distributed logic agrees with single-node |
| **T3 — RunPod / cloud Spark** | RunPod multi-pod or a Spark/Ray cluster (4–16 workers) | M3 at real scale; M5 streaming | hundreds of GB → 100M+ docs | ~$50–300 for a bounded run (spot/community GPUs); set a hard cap | the "at scale" headline number; one bounded run, not a standing cluster |

**Budget discipline:** default to T0/T1 (free/cheap, fully reproducible). T3 is **one bounded run** to produce a headline throughput/cost number, behind an explicit spend cap and `candidateOnly` framing — never a claim of ongoing PB-scale operation. The repo's RunPod MCP + existing `spark-gpu` CI lane make T2/T3 launchable without new infra.

---

## 6. Honest metrics (measured, reproducible)

Every corpus version emits these, generated from the Iceberg snapshot + manifests (so each is reproducible from a hash, not hand-typed):

- **Volume**: raw bytes in, WARC records read, docs extracted, docs kept, **tokens** (and tokens/doc).
- **Dedup**: pre-dedup count, exact-dup rate, near-dup (MinHash LSH) rate, **total dedup rate**; single-node-vs-distributed dedup-rate **agreement** on an overlapping sample (correctness proof).
- **Quality filtering**: per-filter **drop count + survival curve** (how many docs each Gopher/C4/RefinedWeb tagger removed), keep-rate, mean quality, quality histogram — and, where run, the **downstream-loss ablation** (the FineWeb discipline: filter X changed proxy-eval loss by Δ).
- **Decontamination**: **coverage** (fraction of held-out eval shingles checked against the corpus), docs dropped for contamination, per-eval-bank overlap. Honest scope note: catches verbatim/near-verbatim, **not** paraphrase.
- **PII**: redaction counts by type (email/phone/IP/secret).
- **Provenance/lineage**: license breakdown, source/domain breakdown, fraction of rows with full provenance vs `unlicensed`-flagged, fraction capped by the no-provenance cap.
- **Reproducibility**: every shard's `contentSha256`; resolving a pinned **snapshot id** reproduces the exact shard set (CI-asserted); datasheet diffs across versions.
- **Throughput / cost** (T2/T3 only): docs/hour, $/M docs, wall-clock for the bounded run.

All gated by `pipeline/quality_regression.py` so a build that silently degrades **blocks**.

---

## 7. Risks / overclaim guards

- **"Petabyte-scale" overclaim.** Hard rule: claim only the volume actually processed, with the manifest/snapshot to prove it. Frame as "FineWeb/Dolma *methodology* implemented and validated on N GB of real CommonCrawl; designed to scale via Spark/Ray." Never imply a standing PB pipeline.
- **Dedup correctness at scale.** Distributed LSH can disagree with single-node (band/partition boundaries). Mitigation: the **agreement check** in §6 is a required artifact; ship the distributed path only after it matches single-node on an overlap.
- **FineWeb's per-dump-dedup lesson.** Global dedup over-removed and *hurt* FineWeb's downstream eval. Mitigation: dedup **per-dump** by default; any cross-dump dedup must show a downstream-loss ablation before it ships.
- **Decontamination is not a clean bill of health.** Shingle containment misses paraphrase. Mitigation: always report **coverage**, never "decontaminated" as a binary; keep the honest-scope note in every datasheet.
- **Quality filters are value-laden.** Heuristic filters encode bias (FineWeb-Edu's classifier favors academic text). Mitigation: every filter's effect is **ablated and documented**, not assumed; the datasheet lists known limitations.
- **License / PII risk.** CommonCrawl content has mixed licenses and PII. Mitigation: the passport flags `unlicensed`; PII pass redacts + counts; CC's own usage terms respected; no republication of raw scraped content beyond what CC already distributes.
- **Optional-dep drift.** Trafilatura/fastText/Spark/Iceberg are heavy. Mitigation: keep the **stdlib/airgap path** working (regex extract, in-memory dedup, `_catalog.json`) as the CI default; distributed/table paths are additive and tested behind availability checks (the existing `parquet_available()` / `_duckdb_available()` pattern).
- **Don't break what works.** The deterministic, fail-closed, transport-injected discipline is the repo's quality signal. Every new stage must be deterministic and produce a verifiable manifest, or it doesn't ship.

---

## 8. Effort

Assumes one staff-level engineer; "session" ≈ a focused day.

| Milestone | Scope | Effort | Compute |
|---|---|---|---|
| **M1** — real CC processing (stream WARC, full filters, decontam, PII, MDS) | medium-large | **4–6 sessions** | T0/T1 |
| **M2** — Iceberg/Delta catalog + snapshot versioning + lineage | medium | **3–4 sessions** | T1 |
| **M3** — distributed dedup/filter (Spark primary, Ray classifier) + bounded scale run | large | **5–7 sessions** | T2/T3 |
| **M4** — passport/lineage/datasheet differentiator (over the M3 catalog) | medium | **3–4 sessions** | T1 |
| **M5** — streaming (Kafka/Flink) + model-based classifier (stretch) | large | **5–7 sessions** | T3 |

**Critical path to maximum signal:** M1 → M2 → M4 (real CommonCrawl corpus, Iceberg-versioned, with a per-row passport, lineage graph, and a published datasheet + decontamination report) is achievable in ~10–14 sessions and produces the single highest-signal deliverable *without* requiring distributed compute. M3 (Spark/Ray at scale) is the "credible large-scale" stamp and is the right next investment once the single-node story is airtight. M5 is optional polish for the streaming half of the JD.

---

### Appendix — file map (new vs touched)

- **New:** `pipeline/cc/{fetch_sample,warc_stream,extract,run}.py`, `pipeline/filters/` (taggers), `pipeline/decontam.py`, `pipeline/pii.py`, `pipeline/mds.py`, `pipeline/lake/{iceberg_catalog,snapshot,lineage}.py`, `pipeline/datasheet.py`, `pipeline/distributed/{spark_dedup,spark_filter,ray_pipeline}.py`, `pipeline/stream/` (M5).
- **Touched:** `pipeline/fetch/warc.py` (→ streaming), `pipeline/quality_score.py` (→ tagger composition), `pipeline/shard_writer.py` (→ Iceberg write path), `pipeline/corpus_table.py` (→ per-snapshot report), `pipeline/manifest.py` (→ table-level verify), `pretraining/data_passport/passport.py` (→ row-level lineage carrier), `requirements-pipeline.txt` (+ trafilatura, fastText, pyiceberg/deltalake, mosaicml-streaming, pyspark, ray — all optional).
- **Reused as-is (the moat):** `eval/contamination.py`, `provenance_bench/dataset_guard.py`, `agent/{poison_resistant_ingestion,grounded_confidence,corroboration}.py`, `pipeline/dedup/{minhash,vector}.py`, `pipeline/store/{objectstore,s3}.py`, `pipeline/multimodal/shards.py`, `pipeline/quality_regression.py`.

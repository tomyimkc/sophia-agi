# Scale Runbook — processing a real dump

How to run the pipeline (Phases 1–5) over a real multi-GB public corpus and publish numbers.
The same code runs locally (filesystem lake) and on cloud infra (S3/MinIO) — only the
`ObjectStore` adapter changes.

## 1. Get a public dump (pick one)

- **CommonCrawl WARC** — grab a few WARC segments from a monthly crawl
  (`https://data.commoncrawl.org/crawl-data/CC-MAIN-*/warc.paths.gz` → fetch a handful of
  `.warc.gz`). Best fit: `tools/run_scale_pipeline.py` ingests WARC directly.
- **Wikipedia dump** — an HTML/enterprise dump; convert articles to WARC-like records or feed
  as JSONL documents (`{url, content}`) to the dedup→score→shard path.
- **OSCAR / C4 slice** — already cleaned text; skip extraction, run dedup→score→shard.

## 2. Run locally (filesystem lake)

```bash
# one or many .warc(.gz) files
python tools/run_scale_pipeline.py path/to/warcs/ \
    --out-dir ./lake --prefix cc-main-sample --shard-size 5000 --keep-only
```

Outputs a data-lake layout under `./lake/`:

```
lake/
  _seen.db                       # persistent URL+fingerprint seen-set (SqliteSeenSet)
  cc-main-sample/
    _catalog.json                # shard index a training job reads
    part-00000.jsonl             # scored, deduped corpus shard
    part-00000.manifest.json     # per-shard manifest (hash, dedup ratio, quality histogram)
    ...
```

The run prints the headline numbers to publish: input docs, URL + near-dup removed, dedup
ratio, tokens, mean quality, shard count, MB in, and throughput (docs/s).

## 3. Run on cloud infra (S3 / MinIO)

Implement an adapter satisfying `pipeline.store.objectstore.ObjectStore`
(`put/get/exists/list` over boto3) and a `SeenSet` over RocksDB/Redis, then swap them into
`tools/run_scale_pipeline.py`. No pipeline-stage code changes — the stages depend only on the
Protocols. For throughput, fan out by WARC segment (one worker per shard) and push work
through `pipeline.store.queue` (Redis Streams in prod).

## 4. Quality regression gate (daily)

```bash
# first good run -> baseline
python tools/pipeline_stats.py lake/cc-main-sample/part-00000.jsonl --out baseline.summary.json
# subsequent runs fail closed if quality/keep-rate/dedup/tokens regress
python tools/pipeline_stats.py lake/cc-main-sample/part-00001.jsonl --baseline baseline.summary.json
```

## 5. What to report

| Metric | Source |
|--------|--------|
| Docs processed, MB in | scale-pipeline report |
| URL dup % / near-dup % / total dedup ratio | scale-pipeline report |
| Mean quality, keep-rate, quality histogram | `corpus_table` / shard manifests |
| Throughput (docs/s), wall-clock | scale-pipeline report |
| Shard count + catalog | `lake/<prefix>/_catalog.json` |

> **Honesty note.** The committed tests run this path on small fixtures (offline, no cloud).
> Real TB-token numbers require running step 2/3 on actual infra with a real dump — this
> runbook + `tools/run_scale_pipeline.py` are the harness to produce them; the numbers
> themselves should be filled into `RESULTS.md` from an actual run, not asserted in advance.

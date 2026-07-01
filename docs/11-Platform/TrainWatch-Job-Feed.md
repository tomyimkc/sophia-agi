# TrainWatch Universal Job-Feed

design/infra; no capability claim; canClaimAGI stays false.

## Why

TrainWatch (`:8420`) is the cluster's live cam. Until now it had exactly ONE feeder —
`tools/trainwatch_bridge.py` — and it understands ONLY the `tools/train_lora.py` step-log. So a
**cert / bench / judge** run shows *nothing* live at `:8420` while it grinds for an hour; only a
`train_lora` run lights up. The dashboard is blind to most of what the cluster actually does.

`tools/trainwatch_job_feed.py` is the **universal cam**: it generalises the bridge's
"follow a log, feed TrainWatch" loop to **EVERY job type** (cert, bench, judge, train) from
**BOTH sources** (bridge-dispatched and direct-SSH), so every job in progress shows moment-by-moment.

## Design

Three pieces, deliberately split so the parser stays testable and TrainWatch stays optional:

1. **Pure progress parser** — `parse_progress(line) -> {pct, phase, metrics} | None`. Offline,
   deterministic (no clock, no random). It recognises the patterns these jobs emit:
   - HF/tqdm sharded load bar `Loading weights:  34%|..| 60/179` -> `pct` from N/M, phase `loading`
   - generic tqdm `  37%|..| 95/256 [..]` -> `pct`, phase `eval`
   - sophia train step `step S/T ... loss=.. lr=..` -> `pct=S/T`, phase `train`, metrics loss/lr
   - sophia eval `[eval] step S val_loss=..` -> metrics val_loss/train_loss
   - bench-harness step label `>> STEP: A2 — ...` -> phase from the label
   - bench verdict `VERDICT: ... (mean_kl=.., top1=..)` -> phase `done`, metrics parsed from the line
   - done markers `complete (exit 0)` / `training finished` / `saved adapter` -> phase `done`
     (and `complete (exit N>0)` / `FAILED` / `Traceback` -> phase `failed`)
   - anything else -> `None`.
   This is the unit-tested core (`tests/test_trainwatch_job_feed.py`).

2. **Follow loop** — `follow(outfile, name, kind, idle_exit)`. Modelled on `trainwatch_bridge`:
   seek/read the tail, run `parse_progress`, and feed TrainWatch. It registers the run once
   (`trainwatch.init(name, total_steps=100)` — progress is reported as a uniform 0..100 pct so any
   job type gets a sensible bar), calls `run.log({**metrics, "pct": pct}, step=int(pct))` on each
   progress line, and `run.finish("completed"/"failed")` on the done/fail marker or after
   `idle_exit` seconds of silence. The **`import trainwatch` is guarded**: if TrainWatch is not
   installed the feed prints a note and exits 0 — it **never blocks the job**.

3. **Wrapper mode** — `--name X --kind cert -- <cmd...>`. Runs any command, tees its output to a
   log, and follows that log. This is the generic sibling of `scripts/train_with_trainwatch.sh`,
   for **direct-SSH** jobs that don't go through the bridge.

## Both sources feed it

- **Bridge-dispatched jobs** — the spark poller (`tools/github_bridge_poll.py`) already launches
  the job subprocess with stdout -> a temp `outfile`. The poller patch
  (`scripts/spark/2026-06-30-trainwatch-job-feed.patch`) adds, right after that launch, a
  background **sidecar**:
  ```
  python tools/trainwatch_job_feed.py --follow <outfile> --name <cmd id> --kind bench --idle-exit 8
  ```
  It tails the same outfile and feeds TrainWatch. The launch is best-effort and detached: it does
  **not** block the tick loop, and if the feed dies the job is unaffected.

- **Direct-SSH jobs** — run them through the wrapper:
  ```
  python tools/trainwatch_job_feed.py --name cert-v5 --kind cert -- \
    bash scripts/run_local_benchmarks.sh --bench-a
  ```

## CLI

```
python tools/trainwatch_job_feed.py --selftest                      # pure-parser selftest
echo '...' | python tools/trainwatch_job_feed.py --dry-run          # parse stdin, NO trainwatch import
python tools/trainwatch_job_feed.py --follow OUTFILE --name N --kind K   # sidecar (for the poller)
python tools/trainwatch_job_feed.py --name N --kind K -- <cmd...>        # wrapper (direct-SSH)
--idle-exit SECS   # finish the run this many seconds after a done marker / last activity (default 8)
```

## Apply note

The poller patch lives on the feature branch as
`scripts/spark/2026-06-30-trainwatch-job-feed.patch` and is generated against `origin/spark-bridge`.
To enable live feed for bridge jobs:

1. On the Spark, in the bridge checkout, `git apply` the patch to `tools/github_bridge_poll.py`
   (verify `git apply --check` first).
2. Restart the poller so it picks up the new `_start`.

**IMPORTANT: do NOT restart the poller while a job is running** — the in-flight subprocess dies
with the poller's process group, killing the cert/train. **Apply the patch and restart between
jobs only** (when `bridge/STATUS.json` shows `running: null`).

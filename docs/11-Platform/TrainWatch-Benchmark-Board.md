# TrainWatch Benchmark Board

**Status:** design/infra; no capability claim; `canClaimAGI` stays `false`.

`tools/trainwatch_benchmark_board.py` surfaces the **full** benchmark picture — finished AND
unfinished/in-flight — into TrainWatch (`:8420`), so the owner can view everything directly on that
website alongside the trainings. It **mirrors a view**: it does **not** alter the authoritative
`agi-proof/benchmark-results/` JSONs or the failure ledger.

## What it shows (three sources)

1. **Forecast queue (T1–T4).** The pre-registered theory tests from
   `docs/06-Roadmap/Spark-Theory-Test-Forecast.md` are defined as a structured list **inside the
   tool** (the MD is **not** parsed). Each becomes a run named `queue:<key>`:
   - **No result yet** -> `status = pending`, metrics = the forecast band numbers. The whole queue
     shows, so unfinished tests are visible — not only the finished ones.
   - **Matching result JSON present** -> `status = completed`, with the **actuals merged in** (via
     the reused `trainwatch_link_results.extract_run`) on top of the forecast numbers, plus a
     numeric `forecastHit` (1 if every computable actual landed inside its forecast band, else 0).
   - Description = hypothesis + gate + a "forecast vs actual" note.
2. **Finished results.** Result JSON paths or `--glob 'agi-proof/benchmark-results/*.json'` -> a run
   named `result:<stem>` with **every** numeric top-level field surfaced (verdict booleans first as
   1/0, then all remaining numeric fields; generous cap of 40 only to guard an absurd JSON — far
   past the 8-metric default). The textual scope/boundary/verdict notes (`honest_scope`, `boundary`,
   `benchmark`, `task`, `model`, `statsNote`, `caveat`, `claim`, `note`, ...) go in the description
   so they read right on the dashboard.
3. **In-flight jobs.** `--status PATH` (a bridge `STATUS.json`) or `--live` (reads via
   `tools.spark_bridge.read_status`). `status.running` and each `status.pendingCommands` id become
   `job:<id>` runs with `status` running/pending and a metric `isRunning` 1/0; the description
   carries the job `args` / `approvedBy` / `createdBy` / `note` when present. Any real `trainwatch`
   entries already in the status (live trainings the bridge mirrored) are passed through as
   `train:<name>` runs with their step/ETA/loss.

`status` (pending / running / completed) is carried on every run so **unfinished** queries are
visible too — the point of a v1 board that prioritizes completeness of detail over polish.

## Run it on the Spark (TrainWatch installed, `trainwatch serve` up)

```bash
# the full board: queue + every finished result + in-flight bridge jobs, registered into :8420
python tools/trainwatch_benchmark_board.py --queue \
  --glob 'agi-proof/benchmark-results/*.json' --live
trainwatch serve     # -> http://<host>:8420 (over Tailscale too)
```

`register_runs` is the only place `trainwatch` is imported, behind a try-import; it runs only where
TrainWatch is installed. Everywhere else, the extraction is pure + offline + deterministic (no
clock, no random in the pure logic).

## `--dry-run` preview (works anywhere, no trainwatch import)

```bash
python tools/trainwatch_benchmark_board.py --queue \
  --glob 'agi-proof/benchmark-results/baseline-*.json' --dry-run
```

Prints each run's name, status, metrics, and description (pending queue rows + completed result rows
+ job rows), so the board can be previewed from any box before registering on the Spark.

## Self-test

```bash
python tools/trainwatch_benchmark_board.py --selftest
python tests/test_trainwatch_benchmark_board.py
```

Both exercise the pure builders (`board_runs_from_queue`, `board_runs_from_results`,
`board_runs_from_status`) offline: a pending queue item keeps its forecast metrics; a queue item
with a matching result flips to completed with actuals merged + `forecastHit`; a result JSON
surfaces more than 8 numeric fields with notes in the description; a status dict yields `job:` runs
with `isRunning` 1/0; and the builders are deterministic.

## What it does NOT do

- It does not edit any gate / verifier / eval, run any GPU job, or touch the `spark-bridge` branch.
- It does not alter the authoritative `agi-proof/benchmark-results/` JSONs or the failure ledger —
  it only registers a **mirrored view** into TrainWatch.

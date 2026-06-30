#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Benchmark Board — surface the FULL benchmark picture into TrainWatch (:8420).

design/infra; no capability claim; canClaimAGI stays false.

TrainWatch only models TRAINING runs, and ``trainwatch_link_results.py`` mirrors a *single*
finished result JSON as a one-point run. This board generalises that into the WHOLE benchmark
picture — finished AND unfinished/in-flight — so the owner sees everything on the :8420 site:

  1. FORECAST QUEUE — the pre-registered T1–T4 theory tests (defined as a structured list IN this
     tool; we do NOT parse the roadmap MD). Each is a run named ``queue:<key>`` with the forecast
     numbers as metrics; if a matching result JSON exists it flips to ``completed`` with the
     actuals merged in (via ``trainwatch_link_results.extract_run``) plus a ``forecastHit`` 1/0.
     A queue item with no result still shows (``status=pending`` + its forecast) so the owner sees
     the WHOLE queue, not only the finished ones.
  2. FINISHED RESULTS — result JSON paths or ``--glob``. Run named ``result:<stem>`` with ALL
     numeric fields surfaced (verdict booleans first, then every numeric top-level field, generous
     cap) and the textual scope/boundary/verdict notes in the description.
  3. IN-FLIGHT JOBS — an optional bridge ``STATUS.json`` (``--status PATH``) or the live bridge
     (``--live`` -> ``tools.spark_bridge.read_status``). ``status.running`` and each
     ``status.pendingCommands`` id become ``job:<id>`` runs with ``isRunning`` 1/0; any real
     ``trainwatch`` entries already in the status are passed through.

Pure extraction is OFFLINE + DETERMINISTIC (no clock, no random in the pure logic). The actual
``trainwatch.init`` registration runs only where TrainWatch is installed (the Spark), behind a
try-import in ``register_runs``. ``--dry-run`` works anywhere with NO trainwatch import and prints
every run's name/status/metrics/description. This MIRRORS a view; it does not alter the
authoritative ``agi-proof/benchmark-results/`` JSONs or the failure ledger.

Usage:
  python tools/trainwatch_benchmark_board.py --selftest
  python tools/trainwatch_benchmark_board.py --queue --glob 'agi-proof/benchmark-results/*.json' --dry-run
  python tools/trainwatch_benchmark_board.py --queue --status bridge/STATUS.json --dry-run
  python tools/trainwatch_benchmark_board.py --queue --glob 'agi-proof/benchmark-results/*.json' --live
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# REUSE the proven single-result extractor + numeric coercion. NO trainwatch import at module top.
from tools.trainwatch_link_results import _num, extract_run  # noqa: E402

# "Every small detail": surface ALL numeric top-level fields for a finished result, not 8. Generous
# cap guards against an absurd JSON (e.g. a giant numeric map) while still being far past 8.
_RESULT_METRIC_CAP = 40
# Textual fields that describe a finished result — joined into the run description so the owner can
# read the scope / boundary / verdict notes right on the dashboard.
_DESC_FIELDS = (
    "honest_scope", "boundary", "benchmark", "task", "model", "statsNote",
    "caveat", "claim", "note", "discipline", "scheme", "adapter", "base_model",
)


# --------------------------------------------------------------------------------------------------
# 1. FORECAST QUEUE — structured T1–T4 (defined here; the roadmap MD is NOT parsed).
# --------------------------------------------------------------------------------------------------
FORECAST_QUEUE: list[dict] = [
    {
        "key": "T1",
        "name": "NVFP4 mixed-precision cert (down_proj held bf16)",
        "hypothesis": ("Holding the KL-sensitive served projection (down_proj) in bf16 while "
                       "NVFP4-quantizing the rest lifts top1 over the 0.97 floor on an already-"
                       "trained adapter, without breaking mean_kl."),
        "gate": "LowRamGate: mean_kl <= 0.05 AND top1 >= 0.97; protected KL <= 0.10, agree >= 0.95",
        # forecast band keyed by the result-JSON field it predicts, so forecastHit is computable.
        "forecast_metrics": {
            "mean_kl_lo": 0.03, "mean_kl_hi": 0.05,
            "top1_agreement_lo": 0.94, "top1_agreement_hi": 0.96,
            "confidence": 0.60,
        },
        "result_glob": "agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v3.json",
    },
    {
        "key": "T2",
        "name": "CoT faithfulness battery (real local model)",
        "hypothesis": ("A local instruct model's chain-of-thought is partly unfaithful: when an "
                       "answer is swayed by an injected cue, the written reasoning often hides it."),
        "gate": "None — measurement only (rates + bootstrap CIs over FAITH_SEEDS); never a GO",
        "forecast_metrics": {
            "unfaithfulCueUseRate_lo": 0.25, "unfaithfulCueUseRate_hi": 0.45,
            "cueFollowRate_lo": 0.30, "cueFollowRate_hi": 0.55,
            "confidence": 0.55,
        },
        "result_glob": "agi-proof/benchmark-results/*faithfulness*cot*.json",
    },
    {
        "key": "T3",
        "name": "Sophrosyne temperance gate improves decisions",
        "hypothesis": ("The temperance gate (MQ = epsilon - delta over expenditure-vs-demand) "
                       "catches excess and deficiency; with it on, decisions beat a no-gate "
                       "baseline judged by 2 independent families."),
        "gate": "Virtue 2-family: consensus labels (kappa>=0.40) -> paired Delta, bootstrap CI excl. 0",
        "forecast_metrics": {
            "decisionDelta_lo": 0.05, "decisionDelta_hi": 0.15,
            "interJudgeKappa_lo": 0.30, "interJudgeKappa_hi": 0.42,
            "confidence": 0.65,
        },
        "result_glob": "agi-proof/benchmark-results/*sophrosyne*.json",
    },
    {
        "key": "T4",
        "name": "Council vs generalist on a real trained adapter",
        "hypothesis": ("A discipline-routed council (per-seat 3B LoRA + verifier) catches more "
                       "errors on its discipline than one monolithic gate; risk is off-discipline "
                       "regression and that tiny seed corpora make each adapter weak."),
        "gate": "eval_council_vs_monolith (error-catch Delta) then VALIDATED on the on-discipline uplift",
        "forecast_metrics": {
            "onDisciplineDelta_lo": 0.05, "onDisciplineDelta_hi": 0.20,
            "offDisciplineDelta_lo": -0.05, "offDisciplineDelta_hi": 0.0,
            "confidence": 0.50,
        },
        "result_glob": "agi-proof/benchmark-results/council-vs-monolith*.json",
    },
]

# Which actual field each forecast (lo, hi) band predicts, so forecastHit can be computed when the
# result JSON carries that field. Maps actual-field -> (lo_key, hi_key) in forecast_metrics.
_FORECAST_BANDS = {
    "mean_kl": ("mean_kl_lo", "mean_kl_hi"),
    "top1_agreement": ("top1_agreement_lo", "top1_agreement_hi"),
    "unfaithfulCueUseRate": ("unfaithfulCueUseRate_lo", "unfaithfulCueUseRate_hi"),
    "cueFollowRate": ("cueFollowRate_lo", "cueFollowRate_hi"),
    "decisionDelta": ("decisionDelta_lo", "decisionDelta_hi"),
    "interJudgeKappa": ("interJudgeKappa_lo", "interJudgeKappa_hi"),
    "onDisciplineDelta": ("onDisciplineDelta_lo", "onDisciplineDelta_hi"),
    "offDisciplineDelta": ("offDisciplineDelta_lo", "offDisciplineDelta_hi"),
}


def _forecast_hit(forecast: dict, actuals: dict) -> "float | None":
    """1.0 if every computable actual lands inside its forecast band, 0.0 if any is outside, None if
    nothing is computable. Pure + deterministic."""
    seen = False
    for field, (lo_k, hi_k) in _FORECAST_BANDS.items():
        if field not in actuals or lo_k not in forecast or hi_k not in forecast:
            continue
        a = _num(actuals[field])
        lo, hi = _num(forecast[lo_k]), _num(forecast[hi_k])
        if a is None or lo is None or hi is None:
            continue
        seen = True
        if not (lo <= a <= hi):
            return 0.0
    return 1.0 if seen else None


def result_index(paths) -> dict:
    """Pure: build {resolved-glob-or-path : parsed-dict} for the queue to look up actuals against.

    Accepts already-loaded dicts? No — paths are file paths. Caller resolves globs to concrete paths
    first (see ``_resolve_result_paths``); this just reads + parses each into a stem-keyed map AND a
    fullpath-keyed map so queue items can match by their ``result_glob``."""
    index: dict = {}
    for p in paths:
        path = Path(p)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        if isinstance(data, dict):
            index[str(path)] = data
            index[path.stem] = data
    return index


def board_runs_from_queue(queue: list, result_index: dict) -> list:
    """Pure: one run per queue item. ``completed`` if a result matching ``result_glob`` is present in
    ``result_index`` (keyed by resolved path or stem), else ``pending``. Metrics = the forecast
    numbers, PLUS the actuals merged in (via extract_run) when a result exists, PLUS forecastHit 1/0
    when computable. Description = hypothesis + gate + "forecast vs actual". Deterministic."""
    runs: list = []
    for item in queue:
        key = item["key"]
        forecast = dict(item.get("forecast_metrics") or {})
        actual = _match_result(item.get("result_glob"), result_index)
        metrics: dict = dict(forecast)
        if actual is not None:
            status = "completed"
            extracted = extract_run(actual, f"queue:{key}")
            # merge actuals after the forecast so both are visible; actual keys win on collision.
            for mk, mv in extracted["metrics"].items():
                metrics[mk] = mv
            hit = _forecast_hit(forecast, actual)
            if hit is not None:
                metrics["forecastHit"] = hit
            verdict = "forecast vs actual: result present (actuals merged)"
        else:
            status = "pending"
            verdict = "forecast vs actual: pending (no result JSON yet)"
        description = " | ".join([
            f"[{key}] {item.get('name', '')}".strip(),
            f"hypothesis: {item.get('hypothesis', '')}",
            f"gate: {item.get('gate', '')}",
            verdict,
        ])
        runs.append({
            "name": f"queue:{key}",
            "status": status,
            "metrics": metrics,
            "description": description,
        })
    return runs


def _match_result(result_glob, index: dict) -> "dict | None":
    """Pure lookup: find a parsed result in ``index`` for a queue item's ``result_glob``. Matches by
    exact path key, by stem, or — if the glob is a pattern — by any indexed path whose name matches
    the glob's basename pattern. Deterministic (sorted keys)."""
    if not result_glob:
        return None
    g = str(result_glob)
    if g in index:
        return index[g]
    stem = Path(g).stem
    if "*" not in stem and "?" not in stem and stem in index:
        return index[stem]
    # pattern match on basename against indexed full paths (deterministic by sorted key)
    from fnmatch import fnmatch
    base_pat = Path(g).name
    for k in sorted(index):
        if "/" in k or k.endswith(".json"):  # a full-path key
            if fnmatch(Path(k).name, base_pat):
                return index[k]
    return None


def board_runs_from_results(paths) -> list:
    """Pure: one ``result:<stem>`` run per result JSON path, with ALL numeric fields surfaced.

    Order: verdict booleans first (passed/validated/canClaimAGI via extract_run's preference), then
    EVERY numeric top-level field (not capped at 8 — "every small detail"; generous cap of 40 only
    to guard an absurd JSON). Description = the textual scope/boundary/verdict fields present,
    joined. Deterministic (insertion order)."""
    runs: list = []
    for p in paths:
        path = Path(p)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"  skip {p}: not a JSON object", file=sys.stderr)
            continue
        runs.append(_result_run(data, path.stem))
    return runs


def _result_run(data: dict, stem: str) -> dict:
    """Pure single-result run with EVERY numeric field (verdicts first), description from notes."""
    # Seed with extract_run's verdict-first + preferred ordering (reuse), then add every remaining
    # numeric top-level field up to the generous cap — surfacing far more than the 8-metric default.
    seeded = extract_run(data, f"result:{stem}")
    metrics: dict = dict(seeded["metrics"])
    for k, v in data.items():
        if len(metrics) >= _RESULT_METRIC_CAP:
            break
        val = _num(v)
        if val is not None and k not in metrics:
            metrics[k] = val
    metrics = dict(list(metrics.items())[:_RESULT_METRIC_CAP])
    parts: list[str] = []
    for f in _DESC_FIELDS:
        v = data.get(f)
        if isinstance(v, str) and v.strip():
            parts.append(f"{f}: {v.strip()}")
    description = " | ".join(parts) if parts else f"finished result: {stem}"
    return {"name": f"result:{stem}", "status": "completed", "metrics": metrics,
            "description": description}


# --------------------------------------------------------------------------------------------------
# 3. IN-FLIGHT JOBS — from a bridge STATUS.json dict.
# --------------------------------------------------------------------------------------------------
def board_runs_from_status(status: dict) -> list:
    """Pure: in-flight + pending bridge jobs as runs, plus pass-through of any real trainwatch entries.

    - ``status.running`` (a job id or null) -> ``job:<id>`` status "running", metric isRunning=1.
    - each ``status.pendingCommands`` (id strings or {id,...} dicts) -> ``job:<id>`` status
      "pending", metric isRunning=0; description from the command's args/approvedBy when available.
    - each ``status.trainwatch`` entry (a real training the bridge mirrored) -> a ``train:<name>``
      run carrying its current/total steps + latest metrics, status from its own field.
    Deterministic (no clock/random)."""
    runs: list = []
    status = status or {}

    running = status.get("running")
    if running not in (None, "", "null"):
        rid = running if isinstance(running, str) else (running.get("id") if isinstance(running, dict) else str(running))
        meta = running if isinstance(running, dict) else {}
        runs.append({
            "name": f"job:{rid}",
            "status": "running",
            "metrics": {"isRunning": 1.0},
            "description": _job_desc(meta, "running"),
        })

    for cmd in status.get("pendingCommands") or []:
        if isinstance(cmd, dict):
            cid = cmd.get("id", "?")
            meta = cmd
        else:
            cid = str(cmd)
            meta = {}
        runs.append({
            "name": f"job:{cid}",
            "status": "pending",
            "metrics": {"isRunning": 0.0},
            "description": _job_desc(meta, "pending"),
        })

    for tw in status.get("trainwatch") or []:
        if not isinstance(tw, dict):
            continue
        name = tw.get("name") or "?"
        metrics: dict = {}
        cs, ts = _num(tw.get("current_step")), _num(tw.get("total_steps"))
        if cs is not None:
            metrics["current_step"] = cs
        if ts is not None:
            metrics["total_steps"] = ts
        eta = _num(tw.get("eta_seconds"))
        if eta is not None:
            metrics["eta_seconds"] = eta
        for mk, mv in (tw.get("latest_metrics") or {}).items():
            val = _num(mv)
            if val is not None:
                metrics[mk] = val
        st = (tw.get("status") or "running").lower()
        status_norm = "running" if st in ("running", "training", "active") else (
            "completed" if st in ("completed", "done", "finished") else st)
        runs.append({
            "name": f"train:{name}",
            "status": status_norm,
            "metrics": metrics,
            "description": f"live training mirrored by the bridge | status={tw.get('status')}"
                           + (f" | {tw.get('latest_metrics')}" if tw.get("latest_metrics") else ""),
            "steps": {"current": tw.get("current_step"), "total": tw.get("total_steps")},
        })

    return runs


def _job_desc(meta: dict, state: str) -> str:
    parts = [f"bridge job ({state})"]
    if meta.get("args"):
        parts.append(f"args: {meta['args']}")
    if meta.get("approvedBy"):
        parts.append(f"approvedBy: {meta['approvedBy']}")
    if meta.get("createdBy"):
        parts.append(f"createdBy: {meta['createdBy']}")
    if meta.get("note"):
        parts.append(f"note: {meta['note']}")
    return " | ".join(parts)


# --------------------------------------------------------------------------------------------------
# Path resolution (impure: touches the filesystem) + registration (Spark-only, guarded import).
# --------------------------------------------------------------------------------------------------
def _resolve_result_paths(paths, glob_pat) -> list:
    out = list(paths or [])
    if glob_pat:
        out += sorted(globmod.glob(glob_pat))
    # de-dup, preserve order
    seen: set = set()
    uniq: list = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _resolve_queue_index(queue: list) -> dict:
    """Resolve each queue item's result_glob to concrete files, then build a result_index."""
    paths: list = []
    for item in queue:
        g = item.get("result_glob")
        if g:
            paths += sorted(globmod.glob(g))
    return result_index(paths)


def register_runs(runs: list) -> int:
    """Register each run in TrainWatch (Spark-only; trainwatch must be importable). Guarded import —
    this is the ONLY place trainwatch is touched, and never at module top."""
    try:
        import trainwatch
    except Exception as e:  # noqa: BLE001
        print(f"trainwatch not importable here ({e}). Run this on the Spark, or use --dry-run.",
              file=sys.stderr)
        return 2
    n = 0
    for r in runs:
        steps = r.get("steps") or {}
        total = steps.get("total") or 1
        run = trainwatch.init(name=r["name"], description=r.get("description", ""),
                              total_steps=total)
        if r["metrics"]:
            cur = steps.get("current") or 1
            run.log(r["metrics"], step=cur)
        run.finish(r["status"])
        n += 1
        print(f"  registered {r['name']}  status={r['status']}  metrics={r['metrics']}")
    print(f"registered {n} run(s) into TrainWatch (:8420)")
    return 0


def _print_dry(runs: list) -> None:
    for r in runs:
        print(f"\n{r['name']}  [{r['status']}]")
        print(f"  metrics: {r['metrics']}")
        print(f"  desc:    {r['description']}")
    print(f"\n(dry-run) {len(runs)} run(s) — re-run without --dry-run on the Spark to register")


def build_board(args) -> list:
    """Impure orchestration: resolve filesystem inputs, then call the pure builders."""
    runs: list = []
    if args.queue:
        runs += board_runs_from_queue(FORECAST_QUEUE, _resolve_queue_index(FORECAST_QUEUE))
    result_paths = _resolve_result_paths(args.paths, args.glob)
    if result_paths:
        runs += board_runs_from_results(result_paths)
    status = None
    if args.status:
        try:
            status = json.loads(Path(args.status).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"  could not read --status {args.status}: {e}", file=sys.stderr)
    elif args.live:
        try:
            from tools.spark_bridge import read_status
            status = read_status()
        except Exception as e:  # noqa: BLE001
            print(f"  --live read_status failed ({e})", file=sys.stderr)
    if status:
        runs += board_runs_from_status(status)
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="finished result JSON file(s)")
    ap.add_argument("--queue", action="store_true", help="include the forecast queue (T1–T4)")
    ap.add_argument("--glob", default=None, help="glob of finished result JSONs")
    ap.add_argument("--status", default=None, help="path to a bridge STATUS.json (in-flight jobs)")
    ap.add_argument("--live", action="store_true",
                    help="read live bridge status via tools.spark_bridge.read_status")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each run's name/status/metrics/description; NO trainwatch import")
    ap.add_argument("--selftest", action="store_true", help="offline self-test of the pure builders")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    runs = build_board(args)
    if not runs:
        ap.error("nothing to show — pass --queue and/or result paths/--glob and/or --status/--live")
    if args.dry_run:
        _print_dry(runs)
        return 0
    return register_runs(runs)


def _selftest() -> int:
    # queue item with NO result -> pending + forecast metrics present
    qr = board_runs_from_queue(FORECAST_QUEUE, {})
    assert all(r["status"] == "pending" for r in qr), "empty index -> all pending"
    t1 = next(r for r in qr if r["name"] == "queue:T1")
    assert t1["metrics"]["mean_kl_lo"] == 0.03 and "hypothesis:" in t1["description"]

    # queue item WITH a matching result -> completed + actuals merged + forecastHit
    fake_v3 = {"mean_kl": 0.045082, "top1_agreement": 0.90625, "passed": False}
    idx = {"agi-proof/benchmark-results/certify-lowram-olmoe-nvfp4-v3.json": fake_v3,
           "certify-lowram-olmoe-nvfp4-v3": fake_v3}
    qr2 = board_runs_from_queue(FORECAST_QUEUE, idx)
    t1b = next(r for r in qr2 if r["name"] == "queue:T1")
    assert t1b["status"] == "completed"
    assert t1b["metrics"]["mean_kl"] == 0.045082 and t1b["metrics"]["mean_kl_lo"] == 0.03
    # top1 0.90625 is OUTSIDE forecast band [0.94, 0.96] -> forecastHit 0
    assert t1b["metrics"]["forecastHit"] == 0.0

    # a result JSON -> all numeric fields surfaced (more than 8) + description from notes
    res = {"discipline": "biology", "n": 9, "nBad": 4, "nGood": 5, "recallOnBad": 1.0,
           "passRateOnGood": 1.0, "minRecall": 0.9, "minPass": 0.9, "floorMet": True,
           "extra1": 1, "extra2": 2, "caveat": "self-consistency on a fresh set, NOT blind"}
    rr = _result_run(res, "biology-verifier-v2")
    assert len(rr["metrics"]) > 8, f"want >8 numeric fields, got {len(rr['metrics'])}"
    assert rr["metrics"]["floorMet"] == 1.0  # bool verdict -> numeric
    assert "caveat:" in rr["description"]

    # status dict with running + pending -> job: runs with isRunning 1/0
    st = {"running": "cmd-A", "pendingCommands": ["cmd-B", {"id": "cmd-C", "args": "--bench-a"}]}
    sr = board_runs_from_status(st)
    by = {r["name"]: r for r in sr}
    assert by["job:cmd-A"]["status"] == "running" and by["job:cmd-A"]["metrics"]["isRunning"] == 1.0
    assert by["job:cmd-B"]["status"] == "pending" and by["job:cmd-B"]["metrics"]["isRunning"] == 0.0
    assert "args: --bench-a" in by["job:cmd-C"]["description"]

    # trainwatch pass-through
    st2 = {"running": None, "trainwatch": [
        {"name": "olmoe-v5", "current_step": 100, "total_steps": 220, "eta_seconds": 600,
         "status": "running", "latest_metrics": {"loss": 0.54}}]}
    sr2 = board_runs_from_status(st2)
    tw = next(r for r in sr2 if r["name"] == "train:olmoe-v5")
    assert tw["status"] == "running" and tw["metrics"]["loss"] == 0.54
    assert tw["metrics"]["current_step"] == 100.0

    # determinism: same inputs -> identical output
    assert board_runs_from_queue(FORECAST_QUEUE, idx) == board_runs_from_queue(FORECAST_QUEUE, idx)
    assert board_runs_from_status(st) == board_runs_from_status(st)

    print("ok trainwatch_benchmark_board selftest (queue pending/completed + results + jobs + det.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

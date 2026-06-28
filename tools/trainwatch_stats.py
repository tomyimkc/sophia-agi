#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Rich, at-a-glance TrainWatch stats — powers the ``/trainwatch`` Claude Code command.

Pure stdlib (sqlite3 + urllib) so it runs anywhere with no venv. Works **locally** (reads the
TrainWatch store at ``$TRAINWATCH_HOME``, default ``~/.trainwatch/runs.db``) **and remotely**
(fetches the same data from a TrainWatch ``serve`` HTTP API over the network/Tailscale), so the
``/trainwatch`` command works in a Claude session on any machine — including remote-control ones.

Source resolution (first match wins):
  1. ``--url`` / ``$TRAINWATCH_URL``         → remote HTTP API
  2. local ``~/.trainwatch/runs.db`` exists  → local sqlite (fast, most accurate)
  3. otherwise                               → DEFAULT_REMOTE (the Spark on the tailnet)

Time-related fields are foregrounded.

    python3 trainwatch_stats.py                 # all runs, compact
    python3 trainwatch_stats.py <name|id>       # one run, full detail
    TRAINWATCH_URL=http://spark-2f2d:8420 python3 trainwatch_stats.py   # from a remote box
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_REMOTE = os.environ.get("TRAINWATCH_DEFAULT_URL", "http://spark-2f2d:8420")
DB = Path(os.environ.get("TRAINWATCH_HOME", Path.home() / ".trainwatch")) / "runs.db"


# ── data sources ───────────────────────────────────────────────────────────
class SqliteSource:
    kind = "local"

    def __init__(self, path):
        self.c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.c.row_factory = sqlite3.Row

    def runs(self):
        return [dict(r) for r in self.c.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT 50")]

    def series(self, rid):
        out = {}
        for r in self.c.execute("SELECT step, metrics FROM metrics WHERE run_id=? ORDER BY step", (rid,)):
            for k, v in json.loads(r["metrics"]).items():
                try:
                    out.setdefault(k, []).append((r["step"], float(v)))
                except (TypeError, ValueError):
                    pass
        return out

    def rate(self, rid, run, tail=30):
        """Median per-step wall-clock from recent rows; ignores burst-backfill (dt≈0) pairs."""
        rows = [r for r in self.c.execute(
            "SELECT step, logged_at FROM metrics WHERE run_id=? ORDER BY id DESC LIMIT ?",
            (rid, tail))][::-1]
        if len(rows) < 2:
            return None, (rows[-1]["logged_at"] if rows else None)
        per = []
        for a, b in zip(rows, rows[1:]):
            ds, dt = b["step"] - a["step"], b["logged_at"] - a["logged_at"]
            if ds > 0 and dt > 1.0:
                per.append(dt / ds)
        per.sort()
        return (per[len(per) // 2] if per else None), rows[-1]["logged_at"]


class HttpSource:
    kind = "remote"

    def __init__(self, base):
        self.base = base.rstrip("/")

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=6) as r:
            return json.load(r)

    def runs(self):
        return self._get("/api/runs")

    def series(self, rid):
        s = self._get(f"/api/run/{rid}").get("series", {})
        return {k: [(p["step"], p["value"]) for p in v] for k, v in s.items()}

    def rate(self, rid, run):
        # No per-row timestamps over the API → use the server-maintained step time (accurate for
        # runs logged live from the start). last-logged time isn't exposed, so staleness is None.
        ema, sp = run.get("ema_step_time"), run.get("speed")
        return (ema or (1.0 / sp if sp else None)), None


def _source(url):
    if url:
        return HttpSource(url)
    if DB.exists():
        return SqliteSource(DB)
    return HttpSource(DEFAULT_REMOTE)


# ── formatting ─────────────────────────────────────────────────────────────
def _dur(s):
    if s is None:
        return "—"
    s = int(max(s, 0))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def _clock(ts):
    return time.strftime("%m-%d %H:%M:%S", time.localtime(ts)) if ts else "—"


def _eta(run, rate):
    total, cur = run.get("total_steps"), run.get("current_step") or 0
    if run.get("status") != "running":
        return 0
    if rate and total:
        return max(total - cur, 0) * rate
    return run.get("eta_seconds") or 0


def _line(src, run, now):
    total, cur = run.get("total_steps"), run.get("current_step") or 0
    pct = f"{100*cur/total:4.0f}%" if total else "  — "
    elapsed = (run.get("finished_at") or now) - run["started_at"]
    rate, last = src.rate(run["id"], run)
    stale = (now - last) if last else None
    lm = json.loads(run.get("latest_metrics") or "{}") if isinstance(run.get("latest_metrics"), str) else (run.get("latest_metrics") or {})
    keys = [k for k in ("loss", "val_loss", "lr") if k in lm] or list(lm)[:3]
    mstr = " ".join(f"{k}={lm[k]:.4g}" for k in keys)
    flag = ""
    if run.get("status") == "running" and stale and stale > max(900, 6 * (rate or 150)):
        flag = f"  ⚠stalled {_dur(stale)}"
    return (f"{run['id'][:8]}  {run['name'][:22]:22}  {run.get('status',''):9}  {pct}  "
            f"step {cur}/{total or '?'}  elapsed {_dur(elapsed)}  ETA {_dur(_eta(run, rate))}  {mstr}{flag}")


def _detail(src, run, now):
    total, cur = run.get("total_steps"), run.get("current_step") or 0
    elapsed = (run.get("finished_at") or now) - run["started_at"]
    rate, last = src.rate(run["id"], run)
    out = [f"━━ {run['name']}  [{run['id'][:8]}]  {run.get('status','').upper()} ━━"]
    if run.get("description"):
        out.append(f"  {run['description']}")
    out += ["  ── time ──",
            f"   started   : {_clock(run['started_at'])}"]
    if last:
        out.append(f"   last step : {_clock(last)}  ({_dur(now-last)} ago)")
    if run.get("finished_at"):
        out.append(f"   finished  : {_clock(run['finished_at'])}")
    out.append(f"   elapsed   : {_dur(elapsed)}")
    out.append(f"   ETA       : {_dur(_eta(run, rate))}" + ("" if run.get('status') == 'running' else "  (done)"))
    if rate:
        out.append(f"   step rate : {rate:.1f}s/step  ({3600/rate:.0f} steps/hr)")
    out.append("  ── progress ──")
    bn = int(20 * cur / total) if total else 0
    out.append(f"   {('█'*bn + '░'*(20-bn)) if total else '—'}  step {cur}/{total or '?'}"
               + (f"  ({100*cur/total:.1f}%)" if total else ""))
    out.append("  ── metrics ──")
    series = src.series(run["id"])
    for k in sorted(series, key=lambda x: (x not in ("loss", "val_loss", "lr"), x)):
        vals = [v for _, v in series[k]]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        best = lo if ("loss" in k or "err" in k) else hi
        trend = "↓" if len(vals) > 1 and vals[-1] < vals[0] else ("↑" if len(vals) > 1 and vals[-1] > vals[0] else "→")
        out.append(f"   {k:14} latest {vals[-1]:.4g}  best {best:.4g}  range [{lo:.4g}, {hi:.4g}]  {trend} n={len(vals)}")
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    url = None
    args = []
    for a in argv:
        if a.startswith("--url="):
            url = a.split("=", 1)[1]
        elif a == "--url" and argv.index(a) + 1 < len(argv):
            url = argv[argv.index(a) + 1]
        elif a != url:
            args.append(a)
    url = url or os.environ.get("TRAINWATCH_URL")
    try:
        src = _source(url)
        runs = src.runs()
    except Exception as e:
        where = url or (str(DB) if DB.exists() else DEFAULT_REMOTE)
        print(f"TrainWatch unreachable at {where}: {type(e).__name__}: {e}\n"
              f"(start it with `trainwatch serve` on the host, or set TRAINWATCH_URL)")
        return 1
    if not runs:
        print(f"No runs recorded yet ({src.kind}). Start one: tools/trainwatch_bridge.py <log> --name <n>")
        return 0
    now = time.time()
    query = next((a for a in args if a.strip()), None)
    if query:
        m = [r for r in runs if r["id"].startswith(query) or query.lower() in r["name"].lower()]
        if not m:
            print(f"No run matching '{query}'. Known: " + ", ".join(r["name"] for r in runs[:10]))
            return 1
        print(_detail(src, m[0], now))
        return 0
    running = [r for r in runs if r.get("status") == "running"]
    done = [r for r in runs if r.get("status") != "running"]
    tag = "" if src.kind == "local" else f"  [remote {src.base}]"
    print(f"TrainWatch — {len(running)} running, {len(done)} finished{tag}")
    if running:
        print("\nRUNNING:")
        for r in running:
            print("  " + _line(src, r, now))
    if done:
        print("\nFINISHED (recent):")
        for r in done[:8]:
            print("  " + _line(src, r, now))
    print("\nDetail: /trainwatch <name>   (e.g. /trainwatch olmoe-qat-v3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Rich, at-a-glance TrainWatch stats — powers the ``/trainwatch`` Claude Code command.

Pure stdlib (sqlite3 only) so it runs anywhere without a venv. Reads the TrainWatch store at
``$TRAINWATCH_HOME`` (default ``~/.trainwatch/runs.db``) and prints time/ETA/progress + a
per-metric breakdown (latest, best, range, trend). Time-related fields are foregrounded.

    python3 tools/trainwatch_stats.py            # all runs, compact
    python3 tools/trainwatch_stats.py <name|id>  # one run, full detail (recomputes live ETA)
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

DB = Path(os.environ.get("TRAINWATCH_HOME", Path.home() / ".trainwatch")) / "runs.db"


def _fmt_dur(s):
    if s is None:
        return "—"
    s = int(max(s, 0))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def _fmt_clock(ts):
    if not ts:
        return "—"
    return time.strftime("%m-%d %H:%M:%S", time.localtime(ts))


def _conn():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _live_rate(c, run_id, tail=30):
    """Real wall-clock s/step: median per-step time over recent consecutive metric rows.

    Uses the MEDIAN of per-pair (dt/dstep) over pairs with dt>1s, so a burst backfill (many
    rows sharing one timestamp, dt≈0) doesn't skew the estimate — only genuinely live-spaced
    steps contribute. Returns (s_per_step, last_logged_at)."""
    rows = [r for r in c.execute(
        "SELECT step, logged_at FROM metrics WHERE run_id=? ORDER BY id DESC LIMIT ?",
        (run_id, tail))][::-1]
    if len(rows) < 2:
        return None, (rows[-1]["logged_at"] if rows else None)
    per = []
    for a, b in zip(rows, rows[1:]):
        dstep, dt = b["step"] - a["step"], b["logged_at"] - a["logged_at"]
        if dstep > 0 and dt > 1.0:
            per.append(dt / dstep)
    per.sort()
    rate = per[len(per) // 2] if per else None
    return rate, rows[-1]["logged_at"]


def _metric_series(c, run_id):
    series = {}
    for r in c.execute("SELECT step, metrics FROM metrics WHERE run_id=? ORDER BY step", (run_id,)):
        for k, v in json.loads(r["metrics"]).items():
            try:
                series.setdefault(k, []).append((r["step"], float(v)))
            except (TypeError, ValueError):
                pass
    return series


def _run_line(c, r, now):
    total, cur = r["total_steps"], r["current_step"] or 0
    pct = f"{100*cur/total:4.0f}%" if total else "  — "
    elapsed = (r["finished_at"] or now) - r["started_at"]
    rate, last_log = _live_rate(c, r["id"])
    eta = (total - cur) * rate if (rate and total and r["status"] == "running") else (
        0 if r["status"] != "running" else r["eta_seconds"])
    stale = now - last_log if last_log else None
    lm = json.loads(r["latest_metrics"] or "{}")
    keys = [k for k in ("loss", "val_loss", "lr") if k in lm] or list(lm)[:3]
    mstr = " ".join(f"{k}={lm[k]:.4g}" for k in keys)
    flag = ""
    # Flag a stall only well beyond the run's own logging cadence (trainings often print every
    # N steps), so slow-but-healthy runs don't false-positive: 6× step-rate or 15 min.
    stall_after = max(900, 6 * rate) if rate else 900
    if r["status"] == "running" and stale and stale > stall_after:
        flag = f"  ⚠stalled {_fmt_dur(stale)}"
    return (f"{r['id'][:8]}  {r['name'][:22]:22}  {r['status']:9}  {pct}  "
            f"step {cur}/{total or '?'}  elapsed {_fmt_dur(elapsed)}  ETA {_fmt_dur(eta)}  "
            f"{mstr}{flag}")


def _detail(c, r, now):
    total, cur = r["total_steps"], r["current_step"] or 0
    elapsed = (r["finished_at"] or now) - r["started_at"]
    rate, last_log = _live_rate(c, r["id"])
    eta = (total - cur) * rate if (rate and total and r["status"] == "running") else 0
    stale = now - last_log if last_log else None
    out = [f"━━ {r['name']}  [{r['id'][:8]}]  {r['status'].upper()} ━━"]
    if r["description"]:
        out.append(f"  {r['description']}")
    out.append("  ── time ──")
    out.append(f"   started   : {_fmt_clock(r['started_at'])}")
    out.append(f"   last step : {_fmt_clock(last_log)}" + (f"  ({_fmt_dur(stale)} ago)" if stale else ""))
    if r["finished_at"]:
        out.append(f"   finished  : {_fmt_clock(r['finished_at'])}")
    out.append(f"   elapsed   : {_fmt_dur(elapsed)}")
    out.append(f"   ETA       : {_fmt_dur(eta)}" + ("  (done)" if r["status"] != "running" else ""))
    if rate:
        out.append(f"   step rate : {rate:.1f}s/step  ({3600/rate:.0f} steps/hr)")
    out.append("  ── progress ──")
    bar_n = int(20 * cur / total) if total else 0
    out.append(f"   {('█'*bar_n + '░'*(20-bar_n)) if total else '—'}  "
               f"step {cur}/{total or '?'}" + (f"  ({100*cur/total:.1f}%)" if total else ""))
    out.append("  ── metrics ──")
    series = _metric_series(c, r["id"])
    for k in sorted(series, key=lambda x: (x not in ("loss", "val_loss", "lr"), x)):
        vals = [v for _, v in series[k]]
        if not vals:
            continue
        latest = vals[-1]
        lo, hi = min(vals), max(vals)
        best = lo if ("loss" in k or "err" in k) else hi
        trend = "↓" if len(vals) > 1 and vals[-1] < vals[0] else ("↑" if len(vals) > 1 and vals[-1] > vals[0] else "→")
        out.append(f"   {k:14} latest {latest:.4g}  best {best:.4g}  "
                   f"range [{lo:.4g}, {hi:.4g}]  {trend} n={len(vals)}")
    return "\n".join(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not DB.exists():
        print(f"No TrainWatch store at {DB}. Start a run first (tools/trainwatch_bridge.py).")
        return 0
    now = time.time()
    c = _conn()
    runs = c.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 50").fetchall()
    if not runs:
        print("No runs recorded yet.")
        return 0
    query = argv[0].strip() if argv and argv[0].strip() else None
    if query:
        match = [r for r in runs if r["id"].startswith(query) or query.lower() in r["name"].lower()]
        if not match:
            print(f"No run matching '{query}'. Known: " + ", ".join(r["name"] for r in runs[:10]))
            return 1
        print(_detail(c, match[0], now))
        return 0
    running = [r for r in runs if r["status"] == "running"]
    done = [r for r in runs if r["status"] != "running"]
    print(f"TrainWatch — {len(running)} running, {len(done)} finished   (dashboard: trainwatch serve :8420)")
    if running:
        print("\nRUNNING:")
        for r in running:
            print("  " + _run_line(c, r, now))
    if done:
        print("\nFINISHED (recent):")
        for r in done[:8]:
            print("  " + _run_line(c, r, now))
    print("\nDetail: /trainwatch <name>   (e.g. /trainwatch olmoe-qat-v3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

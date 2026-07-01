#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Universal live job-feed — show EVERY cluster job moment-by-moment in TrainWatch (:8420).

design/infra; no capability claim; canClaimAGI stays false.

TrainWatch's only live cam today is ``tools/trainwatch_bridge.py``, which understands ONLY the
``train_lora.py`` step-log. So a cert / bench / judge run shows NOTHING live at :8420 while it
runs for an hour. This tool generalises the bridge's "follow a log, feed TrainWatch" loop to ANY
job type and BOTH sources:

  * BRIDGE-dispatched jobs — the spark poller (``tools/github_bridge_poll.py``) launches the job
    subprocess with stdout -> a temp outfile; a sidecar ``--follow <outfile> --kind bench`` tails
    that same file and feeds TrainWatch.
  * DIRECT-SSH jobs — the wrapper mode ``--name X --kind cert -- <cmd...>`` runs any command, tees
    its output to a log, and follows it (the generic sibling of ``scripts/train_with_trainwatch.sh``).

The PROGRESS PARSER (``parse_progress``) is PURE / OFFLINE / DETERMINISTIC (no clock, no random) and
unit-tested. It recognises the patterns cert/bench/judge/train jobs actually emit (HF/tqdm bars,
sophia step/eval lines, the bench-harness ``>> STEP:`` labels, the ``VERDICT:`` line, and done
markers). The ``trainwatch.init/run.log/run.finish`` calls live behind a GUARDED import: if
TrainWatch is absent the tool prints a note and exits 0 — it NEVER blocks the job.

Usage:
  python tools/trainwatch_job_feed.py --selftest
  echo 'Loading weights:  34%|x| 60/179' | python tools/trainwatch_job_feed.py --dry-run
  python tools/trainwatch_job_feed.py --follow /tmp/job.out --name cmd-123 --kind bench --idle-exit 8
  python tools/trainwatch_job_feed.py --name cert-v5 --kind cert -- bash scripts/run_local_benchmarks.sh --bench-a
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# --- pure progress patterns (deterministic; no clock/random) ----------------------------------

# HF / sharded-load tqdm bar with an explicit label:  "Loading weights:  34%|... | 60/179"
_LOAD = re.compile(r"(?P<label>[A-Za-z][A-Za-z ./_-]*?):\s*\d+%\|.*?\|\s*(?P<n>\d+)\s*/\s*(?P<m>\d+)")
# generic tqdm bar (no leading label):  "  37%|...| 95/256 [..]"
_TQDM = re.compile(r"^\s*\d+%\|.*?\|\s*(?P<n>\d+)\s*/\s*(?P<m>\d+)")
# sophia train step line:  "epoch 2/2 step 200/220 (90.9%) loss=0.5418 lr=1.93e-06"
_STEP = re.compile(r"step\s+(?P<s>\d+)\s*/\s*(?P<t>\d+).*?loss=(?P<loss>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                   r"(?:.*?lr=(?P<lr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?")
# eval line:  "[eval] step 150 val_loss=1.5012 train_loss=0.5373"
_EVAL = re.compile(r"\[eval\]\s+step\s+(?P<s>\d+)\s+val_loss=(?P<vl>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
                   r"(?:\s+train_loss=(?P<tl>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?))?")
# bench-harness step label:  ">> STEP: A2 — Bench A" / "B2 — Certify .."
_BENCHSTEP = re.compile(r">>\s*STEP:\s*(?P<label>[A-Za-z0-9][^\n]*?)\s*$")
# verdict line with metrics:  "VERDICT: PASS (mean_kl=0.045, top1=0.906)"
_VERDICT = re.compile(r"VERDICT:")
_KV = re.compile(r"(?P<k>mean_kl|top1|top1_agreement|val_loss|loss|kl|pass)\s*=\s*"
                 r"(?P<v>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
# done markers any of these jobs write
_DONE = re.compile(r"complete \(exit 0\)|training finished|saved adapter|"
                   r"\.train_complete|=== certify done|\btraining complete\b")
_FAIL = re.compile(r"complete \(exit [1-9]\d*\)|\bFAILED\b|\berror exit\b|Traceback \(most recent")


def _f(x):
    return float(x) if x is not None else None


def parse_progress(line: str):
    """Pure: map ONE log line to {pct, phase, metrics} or None (unknown line).

    pct is 0..100 (int when from N/M, else None); phase is a short label; metrics is a flat
    numeric dict. Deterministic — no clock, no random. Order matters: the most specific
    sophia/bench patterns win before the generic tqdm fallback.
    """
    if not line:
        return None

    # done / fail markers first (a job's final line)
    if _DONE.search(line):
        return {"pct": 100, "phase": "done", "metrics": {}}
    if _FAIL.search(line):
        return {"pct": None, "phase": "failed", "metrics": {}}

    # sophia train step:  step S/T ... loss=.. lr=..
    m = _STEP.search(line)
    if m:
        s, t = int(m.group("s")), int(m.group("t"))
        pct = round(100.0 * s / t) if t else None
        metrics = {"loss": _f(m.group("loss")), "step": s, "total": t}
        if m.group("lr"):
            metrics["lr"] = _f(m.group("lr"))
        return {"pct": pct, "phase": "train", "metrics": metrics}

    # eval line:  [eval] step S val_loss=..
    e = _EVAL.search(line)
    if e:
        metrics = {"val_loss": _f(e.group("vl")), "step": int(e.group("s"))}
        if e.group("tl"):
            metrics["train_loss"] = _f(e.group("tl"))
        return {"pct": None, "phase": "eval", "metrics": metrics}

    # bench-harness verdict:  VERDICT: ... (mean_kl=.., top1=..)
    if _VERDICT.search(line):
        metrics = {}
        for kv in _KV.finditer(line):
            metrics[kv.group("k")] = _f(kv.group("v"))
        return {"pct": 100, "phase": "done", "metrics": metrics}

    # bench-harness step label:  >> STEP: A2 — ...
    b = _BENCHSTEP.search(line)
    if b:
        label = b.group("label").strip()
        return {"pct": None, "phase": label, "metrics": {}}

    # HF/sharded load bar with a label:  "Loading weights:  34%|..| 60/179"
    ld = _LOAD.search(line)
    if ld:
        n, mm = int(ld.group("n")), int(ld.group("m"))
        pct = round(100.0 * n / mm) if mm else None
        return {"pct": pct, "phase": "loading", "metrics": {"n": n, "total": mm}}

    # generic tqdm bar:  "  37%|..| 95/256 [..]" -> eval phase
    tq = _TQDM.search(line)
    if tq:
        n, mm = int(tq.group("n")), int(tq.group("m"))
        pct = round(100.0 * n / mm) if mm else None
        return {"pct": pct, "phase": "eval", "metrics": {"n": n, "total": mm}}

    return None


# --- follow loop (TrainWatch-side; guarded import) --------------------------------------------

def _try_trainwatch():
    """Return the trainwatch module or None (guarded; module must import without it)."""
    try:
        import trainwatch  # noqa: F401
        return trainwatch
    except Exception:  # noqa: BLE001
        return None


def follow(outfile, name, kind, idle_exit=8.0):
    """Tail ``outfile``, parse each line, and feed TrainWatch live. Never blocks the job.

    Registers one run (``init(name, total_steps=100)`` — we report progress as a 0..100 pct so any
    job type gets a uniform progress bar). On each parsed progress line we ``run.log`` the metrics +
    pct (step=int(pct)); on a done/fail marker, or after ``idle_exit`` seconds of silence following
    any activity, we ``run.finish``. If TrainWatch is absent we print a note and return 0.
    """
    tw = _try_trainwatch()
    if tw is None:
        print(f"[trainwatch_job_feed] trainwatch not installed; nothing to feed for {name!r} "
              f"(job is unaffected)")
        return 0

    path = Path(outfile)
    for _ in range(60):  # tolerate being launched slightly before the outfile exists
        if path.exists():
            break
        time.sleep(1)
    if not path.exists():
        print(f"[trainwatch_job_feed] no such outfile: {path}", file=sys.stderr)
        return 0  # never fail the job over a missing feed

    run = tw.init(name=name, description=f"{kind}: {path.name}", total_steps=100)
    pos = 0
    saw_done = False
    final = "completed"
    last_change = time.time()
    last_pct = 0
    while True:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
        except OSError:
            chunk = ""
        for line in chunk.splitlines():
            prog = parse_progress(line)
            if prog is None:
                continue
            last_change = time.time()
            metrics = dict(prog["metrics"])
            metrics["phase"] = prog["phase"]
            pct = prog["pct"]
            if pct is not None:
                metrics["pct"] = pct
                last_pct = pct
            try:
                run.log({k: v for k, v in metrics.items() if v is not None},
                        step=int(pct) if pct is not None else last_pct)
            except Exception:  # noqa: BLE001
                pass
            if prog["phase"] == "done":
                saw_done = True
                final = "completed"
            elif prog["phase"] == "failed":
                saw_done = True
                final = "failed"
        if saw_done and (time.time() - last_change) > idle_exit:
            break
        # idle-exit even without a done marker (job died / silent), so the feed never hangs forever
        if not saw_done and (time.time() - last_change) > max(idle_exit * 6, 60):
            final = "completed"
            break
        time.sleep(2)

    try:
        run.finish(final)
    except Exception:  # noqa: BLE001
        pass
    print(f"[trainwatch_job_feed] {name}: {final} (kind={kind})")
    return 0


def _wrapper(name, kind, cmd, idle_exit, logpath=None):
    """Run ``cmd``, tee its output to a log, and follow it (direct-SSH jobs)."""
    log = Path(logpath) if logpath else Path(os.environ.get("TW_LOG", f"logs/jobfeed/{name}.log"))
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("")  # fresh log so the follower starts clean

    import threading
    feeder = threading.Thread(target=follow, args=(str(log), name, kind, idle_exit), daemon=True)
    feeder.start()

    with open(log, "a", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()
        rc = proc.wait()
        # emit a done marker the parser recognises so the feed finalises promptly
        fh.write(f"complete (exit {rc})\n")
        fh.flush()

    # give the feeder up to idle_exit+4s to flush the final point before we return
    feeder.join(timeout=idle_exit + 4)
    print(f"[trainwatch_job_feed] wrapped job '{name}' exited rc={rc}")
    return rc


def _selftest() -> int:
    cases = [
        ("Loading weights:  34%|##  | 60/179", 34, "loading"),
        ("  37%|####  | 95/256 [00:10<00:20]", 37, "eval"),
        ("epoch 2/2 step 200/220 (90.9%) loss=0.5418 lr=1.93e-06", 91, "train"),
        ("  [eval] step 150 val_loss=1.5012 train_loss=0.5373", None, "eval"),
        (">> STEP: A2 — Bench A", None, "A2 — Bench A"),
        ("VERDICT: PASS (mean_kl=0.045, top1=0.906)", 100, "done"),
        ("complete (exit 0)", 100, "done"),
        ("training finished (rc=0)", 100, "done"),
        ("saved adapter to checkpoints/v5", 100, "done"),
        ("random noise line", None, None),
    ]
    for line, want_pct, want_phase in cases:
        got = parse_progress(line)
        if want_phase is None:
            assert got is None, f"expected None for {line!r}, got {got}"
            continue
        assert got is not None, f"expected progress for {line!r}"
        assert got["pct"] == want_pct, f"{line!r}: pct {got['pct']} != {want_pct}"
        assert got["phase"] == want_phase, f"{line!r}: phase {got['phase']!r} != {want_phase!r}"
    # VERDICT metric extraction
    v = parse_progress("VERDICT: PASS (mean_kl=0.045, top1=0.906)")
    assert v["metrics"]["mean_kl"] == 0.045 and v["metrics"]["top1"] == 0.906, v
    # step metrics
    s = parse_progress("step 200/220 loss=0.5 lr=1e-6")
    assert s["metrics"]["loss"] == 0.5 and s["metrics"]["lr"] == 1e-6 and s["pct"] == 91, s
    # determinism
    assert parse_progress(cases[0][0]) == parse_progress(cases[0][0])
    print("trainwatch_job_feed selftest: PASS")
    return 0


def _dry_run() -> int:
    """Print parsed progress from stdin. NO trainwatch import (works anywhere)."""
    n = 0
    for line in sys.stdin:
        prog = parse_progress(line.rstrip("\n"))
        if prog is not None:
            print(prog)
            n += 1
    if n == 0:
        print("(no parseable progress lines)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--follow", metavar="OUTFILE", help="sidecar mode: tail OUTFILE and feed TrainWatch")
    ap.add_argument("--name", help="TrainWatch run name (cmd id / run name)")
    ap.add_argument("--kind", default="job", help="job kind: cert | bench | judge | train | job")
    ap.add_argument("--log", default=None, help="wrapper mode: log path to tee into (default logs/jobfeed/<name>.log)")
    ap.add_argument("--idle-exit", type=float, default=8.0,
                    help="finish the run this many seconds after a done marker / last activity")
    ap.add_argument("--selftest", action="store_true", help="run the pure-parser selftest and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print parsed progress from stdin (no trainwatch import)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="wrapper mode: '-- <command...>' to run, tee, and follow")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.dry_run:
        return _dry_run()

    # wrapper mode: everything after '--'
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if cmd:
        if not args.name:
            ap.error("wrapper mode requires --name")
        return _wrapper(args.name, args.kind, cmd, args.idle_exit, args.log)

    # sidecar mode
    if args.follow:
        if not args.name:
            ap.error("--follow requires --name")
        return follow(args.follow, args.name, args.kind, args.idle_exit)

    ap.error("nothing to do: pass --selftest, --dry-run, --follow OUTFILE, or -- <cmd...>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

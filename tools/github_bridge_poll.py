#!/usr/bin/env python3
"""Hermes-side poller for the GitHub-mediated Spark bridge (branch `spark-bridge`).

Runs ON THE SPARK. Talks only to GitHub (git) and localhost trainwatch.

Execution model (default = NON-BLOCKING):
  * At most ONE command runs at a time (GPU exclusivity), started as a background
    subprocess. The tick loop does NOT block on it, so every ~interval seconds it keeps
    publishing bridge/STATUS.json (trainwatch progress stays LIVE during a multi-hour
    train) and re-syncs the branch.
  * When the running job exits, its result (exitCode + capped stdout tail + touched
    artifacts) is written to bridge/results/<id>.json and committed/pushed.
  * `--blocking` reverts to the old synchronous behavior (run-to-completion in-tick).

Safety: only `scripts/run_local_benchmarks.sh` is ever executed, only with allowlisted
flags; `--execute`/`--run-train` require a non-empty `approvedBy`.

Keep it alive across sessions (fully detached, torch venv exported):
  tmux new-session -d -s bridge-poller \
    'cd /home/tomyimkc/sophia-bridge && PYTHON=/home/tomyimkc/sophia-agi/.venv/bin/python \
     exec python3 tools/github_bridge_poll.py --interval 30 \
       --trainwatch http://127.0.0.1:8420/api/runs >> ~/bridge-poll.log 2>&1'
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ALLOWLIST = {"--dry-run", "--bench-a", "--bench-b", "--bench-virtues", "--all", "--execute", "--run-train"}
GATED = {"--execute", "--run-train"}  # require approvedBy
RUNNER = "scripts/run_local_benchmarks.sh"
MAX_TAIL = 64 * 1024  # cap stdout stored in the result

# A command may carry config env (so cert/train run THROUGH the bridge and their result lands in
# bridge/results/). Defense-in-depth: the poller RE-validates against its OWN allowlist + value
# regex and silently drops anything else — it never trusts the command file to set arbitrary env
# (no PATH/LD_*). Mirrors tools/spark_bridge.ENV_ALLOWLIST.
ENV_ALLOWLIST = {
    "SEEDS", "ANSWERS_DIR", "ANSWERS_PREFIX", "JUDGE_OUT_DIR", "JUDGMENTS", "UPLIFT_OUT",
    "JUDGES", "JUDGE_CONFIG", "AUTO_START_JUDGES", "PYTHON",
    "QAT_BASE", "QAT_ADAPTER", "QAT_DATA", "QAT_EPOCHS", "QAT_LAMBDA",
    "KEEP_SUFFIXES", "CERT_NEVAL", "CERT_CALIB", "CERT_OUT",
    "SPARK_HOST", "MAC_HOST", "SPARK_PORT", "MAC_PORT", "SPARK_JUDGE_MODEL", "MAC_JUDGE_MODEL",
    "VIRTUE_JUDGE_A", "VIRTUE_JUDGE_B", "VIRTUE_JUDGE_A_NAME", "VIRTUE_JUDGE_B_NAME",
    "VIRTUE_SUBJECT", "VIRTUE_SEEDS", "THINKING_MODEL", "FAITH_MODEL", "FAITH_SEEDS", "FAITH_BATTERY",
}
_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:@,+= -]{0,200}$")


def _safe_env(cmd):
    """os.environ overlaid with the command's ALLOWLISTED, value-sanitized env (config knobs only)."""
    env = dict(os.environ)
    for k, v in (cmd.get("env") or {}).items():
        if k in ENV_ALLOWLIST and isinstance(v, str) and _ENV_VALUE_RE.match(v):
            env[k] = v
        else:
            sys.stderr.write(f"[bridge] dropping disallowed/unsafe env {k!r} from cmd {cmd.get('id')}\n")
    return env

# In-memory record of the single in-flight job (non-blocking mode). Lost on restart (the
# subprocess dies with the poller's process group), so its command simply has no result and
# is re-picked on the next run.
_RUNNING = None  # dict: id, proc, cmd, started, outfile, before


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args, cwd, check=False):
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)


def _sync(cwd, branch):
    """ff-only sync (avoids re-smudging git-crypt files); rebase then reset as fallbacks."""
    _git("fetch", "origin", branch, cwd=cwd)
    if _git("merge", "--ff-only", f"origin/{branch}", cwd=cwd).returncode == 0:
        return
    if _git("pull", "--rebase", "origin", branch, cwd=cwd).returncode == 0:
        return
    sys.stderr.write("[bridge] ff + rebase failed; last-resort reset --hard\n")
    _git("reset", "--hard", f"origin/{branch}", cwd=cwd)


def _push_with_retry(cwd, branch):
    delay = 2
    for attempt in range(4):
        if _git("push", "origin", branch, cwd=cwd).returncode == 0:
            return
        sys.stderr.write(f"[bridge] push failed (try {attempt+1}); rebasing\n")
        time.sleep(delay); delay *= 2
        _git("pull", "--rebase", "origin", branch, cwd=cwd)
    sys.stderr.write("[bridge] push giving up this tick; retry next tick\n")


def _trainwatch(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (localhost)
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"trainwatch unreachable: {exc}"}


def _artifact_index(root):
    base = root / "agi-proof" / "benchmark-results"
    out = []
    if not base.exists():
        return out
    for p in sorted(base.rglob("*.json")):
        try:
            st = p.stat()
            out.append({"path": str(p.relative_to(root)), "bytes": st.st_size,
                        "mtime": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc)
                            .strftime("%Y-%m-%dT%H:%M:%SZ")})
        except OSError:
            continue
    return out


def _validate(args_str, approved_by):
    toks = args_str.split()
    if not toks:
        return [], "empty args"
    bad = [t for t in toks if t not in ALLOWLIST]
    if bad:
        return [], f"disallowed tokens {bad}; allowlist={sorted(ALLOWLIST)}"
    if any(t in GATED for t in toks) and not approved_by.strip():
        return [], f"{sorted(set(toks) & GATED)} require a non-empty approvedBy"
    return toks, None


def _write_result(root, res):
    (root / "bridge" / "results" / f"{res['id']}.json").write_text(
        json.dumps(res, indent=2) + "\n")


def _finish_result(root, rec, rc):
    out = ""
    try:
        out = Path(rec["outfile"]).read_text(errors="replace")
    except OSError:
        pass
    after = _artifact_index(root)
    touched = [a["path"] for a in after if rec["before"].get(a["path"]) != a["mtime"]]
    return {"id": rec["id"], "args": rec["cmd"].get("args", ""),
            "status": "ok" if rc == 0 else "error", "exitCode": rc,
            "startedAt": rec["started"], "endedAt": _now(),
            "approvedBy": str(rec["cmd"].get("approvedBy", "")),
            "stdoutTail": out[-MAX_TAIL:], "artifactsTouched": touched}


def _start(root, cmd):
    """Launch run_local_benchmarks.sh for `cmd` as a background subprocess; return a record."""
    toks = cmd["_toks"]
    fd, outpath = tempfile.mkstemp(prefix=f"bridge-{cmd.get('id','job')}-", suffix=".out")
    fh = open(fd, "w")
    proc = subprocess.Popen(["bash", RUNNER, *toks], cwd=str(root), stdout=fh,
                            stderr=subprocess.STDOUT, text=True, env=_safe_env(cmd))
    # Universal live job-feed: tail the SAME outfile and feed TrainWatch so EVERY bridge job
    # (cert/bench/judge/train) shows live at :8420, not just train_lora. Best-effort + detached:
    # never block the tick loop, and a dead feed must not fail the job (hence the try/except).
    try:
        subprocess.Popen([sys.executable, "tools/trainwatch_job_feed.py", "--follow", outpath,
                          "--name", str(cmd.get("id", "job")), "--kind", "bench", "--idle-exit", "8"],
                         cwd=str(root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[bridge] job-feed sidecar failed to launch (job unaffected): {exc}\n")
    return {"id": cmd["id"], "proc": proc, "cmd": cmd, "started": _now(),
            "outfile": outpath, "before": {a["path"]: a["mtime"] for a in _artifact_index(root)},
            "_fh": fh}


def _pending(root):
    cmds = sorted((root / "bridge" / "commands").glob("*.json"))
    done = {p.stem for p in (root / "bridge" / "results").glob("*.json")}
    return [p for p in cmds if p.stem not in done]


def _publish(root, trainwatch_url, pending_ids, running_id):
    status = {"updatedAt": _now(), "host": "spark-2f2d",
              "running": running_id, "pendingCommands": pending_ids,
              "trainwatch": _trainwatch(trainwatch_url),
              "artifacts": _artifact_index(root)}
    (root / "bridge" / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")


def _commit_push(root, branch, msg):
    _git("add", "bridge", cwd=root)
    if _git("commit", "-m", msg, cwd=root).returncode == 0:
        _push_with_retry(root, branch)


def tick_nonblocking(root, branch, trainwatch_url):
    global _RUNNING
    _sync(root, branch)
    (root / "bridge" / "commands").mkdir(parents=True, exist_ok=True)
    (root / "bridge" / "results").mkdir(parents=True, exist_ok=True)

    note = ""
    # 1. reap the in-flight job if it finished
    if _RUNNING is not None:
        rc = _RUNNING["proc"].poll()
        if rc is None:
            _publish(root, trainwatch_url, [p.stem for p in _pending(root)], _RUNNING["id"])
            _commit_push(root, branch, f"bridge: status @ {_now()} (running {_RUNNING['id']})")
            return
        try:
            _RUNNING["_fh"].close()
        except Exception:  # noqa: BLE001
            pass
        res = _finish_result(root, _RUNNING, rc)
        _write_result(root, res)
        note = f" + result {res['id']} ({res['status']})"
        sys.stderr.write(f"[bridge] {res['id']} finished rc={rc}\n")
        _RUNNING = None

    # 2. start the next pending job (one at a time)
    started = None
    if _RUNNING is None:
        for cpath in _pending(root):
            try:
                cmd = json.loads(cpath.read_text())
            except Exception as exc:  # noqa: BLE001
                _write_result(root, {"id": cpath.stem, "status": "rejected",
                                     "reason": f"unparseable: {exc}", "endedAt": _now()})
                note += f" + rejected {cpath.stem}"
                continue
            toks, reason = _validate(str(cmd.get("args", "")), str(cmd.get("approvedBy", "")))
            if reason:
                _write_result(root, {"id": cmd.get("id", cpath.stem), "args": cmd.get("args", ""),
                                     "status": "rejected", "reason": reason, "endedAt": _now(),
                                     "exitCode": None})
                note += f" + rejected {cmd.get('id', cpath.stem)}"
                continue
            cmd["_toks"] = toks
            cmd.setdefault("id", cpath.stem)
            _RUNNING = _start(root, cmd)
            started = cmd["id"]
            sys.stderr.write(f"[bridge] started {started}: {cmd.get('args')!r}\n")
            break

    running_id = _RUNNING["id"] if _RUNNING else None
    _publish(root, trainwatch_url, [p.stem for p in _pending(root)], running_id)
    extra = f" + started {started}" if started else ""
    _commit_push(root, branch, f"bridge: status @ {_now()}{note}{extra}")
    sys.stderr.write(f"[bridge] tick @ {_now()} running={running_id}{note}{extra}\n")


def tick_blocking(root, branch, trainwatch_url):
    """Legacy synchronous tick (run-to-completion in-tick); --blocking fallback."""
    _sync(root, branch)
    (root / "bridge" / "commands").mkdir(parents=True, exist_ok=True)
    (root / "bridge" / "results").mkdir(parents=True, exist_ok=True)
    pending = _pending(root)
    _publish(root, trainwatch_url, [p.stem for p in pending], None)
    ran = []
    for cpath in pending:
        try:
            cmd = json.loads(cpath.read_text())
        except Exception as exc:  # noqa: BLE001
            _write_result(root, {"id": cpath.stem, "status": "rejected",
                                 "reason": f"unparseable: {exc}", "endedAt": _now()})
            continue
        toks, reason = _validate(str(cmd.get("args", "")), str(cmd.get("approvedBy", "")))
        cid = cmd.get("id", cpath.stem)
        if reason:
            _write_result(root, {"id": cid, "args": cmd.get("args", ""), "status": "rejected",
                                 "reason": reason, "endedAt": _now(), "exitCode": None})
            continue
        before = {a["path"]: a["mtime"] for a in _artifact_index(root)}
        proc = subprocess.run(["bash", RUNNER, *toks], cwd=str(root),
                              capture_output=True, text=True, env=_safe_env(cmd))
        after = _artifact_index(root)
        _write_result(root, {"id": cid, "args": cmd.get("args", ""),
                             "status": "ok" if proc.returncode == 0 else "error",
                             "exitCode": proc.returncode, "startedAt": _now(), "endedAt": _now(),
                             "approvedBy": str(cmd.get("approvedBy", "")),
                             "stdoutTail": (proc.stdout or "")[-MAX_TAIL:],
                             "stderrTail": (proc.stderr or "")[-4096:],
                             "artifactsTouched": [a["path"] for a in after
                                                  if before.get(a["path"]) != a["mtime"]]})
        ran.append(cid)
    _git("add", "bridge", cwd=root)
    msg = f"bridge: status @ {_now()}" + (f" + {len(ran)} result(s): {', '.join(ran)}" if ran else "")
    if _git("commit", "-m", msg, cwd=root).returncode == 0:
        _push_with_retry(root, branch)


def main():
    ap = argparse.ArgumentParser(description="GitHub-mediated Spark bridge poller")
    ap.add_argument("--repo-dir", default=".", type=Path)
    ap.add_argument("--branch", default="spark-bridge")
    ap.add_argument("--trainwatch", default="http://127.0.0.1:8420/api/runs")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--blocking", action="store_true",
                    help="legacy synchronous mode (run-to-completion in-tick); fallback")
    a = ap.parse_args()
    root = a.repo_dir.resolve()
    tick = tick_blocking if a.blocking else tick_nonblocking
    sys.stderr.write(f"[bridge] poller on {root} branch={a.branch} interval={a.interval}s "
                     f"mode={'blocking' if a.blocking else 'nonblocking'} once={a.once}\n")
    while True:
        try:
            tick(root, a.branch, a.trainwatch)
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"[bridge] tick error: {exc}\n")
        if a.once:
            return 0
        time.sleep(max(5, a.interval))


if __name__ == "__main__":
    raise SystemExit(main())

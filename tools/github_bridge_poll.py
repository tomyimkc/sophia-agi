#!/usr/bin/env python3
"""Hermes-side poller for the GitHub-mediated Spark bridge (branch `spark-bridge`).

Runs ON THE SPARK. Talks only to GitHub (git) and localhost trainwatch. On each
tick it: (1) fast-forwards the bridge branch, (2) publishes bridge/STATUS.json
(trainwatch snapshot + benchmark-artifact index), (3) executes any approved,
allowlisted commands by invoking ONLY scripts/run_local_benchmarks.sh, (4) writes
bridge/results/<id>.json, (5) commits + pushes.

This is a control plane for OWNED-HARDWARE benchmarks only. It cannot run any
binary other than scripts/run_local_benchmarks.sh, and only with these flags:
  --dry-run --bench-a --bench-b --all --execute --run-train
--execute / --run-train additionally require a non-empty `approvedBy`.

Usage (on the Spark, inside the repo checkout):
  git fetch origin spark-bridge && git checkout spark-bridge
  python3 tools/github_bridge_poll.py --interval 30 \
      --trainwatch http://127.0.0.1:8420/api/runs
  # one tick only:
  python3 tools/github_bridge_poll.py --once
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ALLOWLIST = {"--dry-run", "--bench-a", "--bench-b", "--all", "--execute", "--run-train"}
GATED = {"--execute", "--run-train"}  # require approvedBy
RUNNER = "scripts/run_local_benchmarks.sh"
MAX_TAIL = 64 * 1024  # cap stdout stored in the result


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)


def _push_with_retry(cwd: Path, branch: str) -> None:
    delay = 2
    for attempt in range(4):
        r = _git("push", "origin", branch, cwd=cwd, check=False)
        if r.returncode == 0:
            return
        sys.stderr.write(f"[bridge] push failed (try {attempt+1}): {r.stderr.strip()}\n")
        time.sleep(delay)
        delay *= 2
        # someone else may have pushed; rebase our bridge commits on top
        _git("pull", "--rebase", "origin", branch, cwd=cwd, check=False)
    sys.stderr.write("[bridge] push giving up this tick; will retry next tick\n")


def _trainwatch_snapshot(url: str) -> object:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 (localhost)
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"error": f"trainwatch unreachable: {exc}"}


def _artifact_index(root: Path) -> list[dict]:
    base = root / "agi-proof" / "benchmark-results"
    out: list[dict] = []
    if not base.exists():
        return out
    for p in sorted(base.rglob("*.json")):
        try:
            st = p.stat()
            out.append({
                "path": str(p.relative_to(root)),
                "bytes": st.st_size,
                "mtime": _dt.datetime.fromtimestamp(st.st_mtime, _dt.timezone.utc)
                    .strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
        except OSError:
            continue
    return out


def _validate(args_str: str, approved_by: str) -> tuple[list[str], str | None]:
    toks = args_str.split()
    if not toks:
        return [], "empty args"
    bad = [t for t in toks if t not in ALLOWLIST]
    if bad:
        return [], f"disallowed tokens {bad}; allowlist={sorted(ALLOWLIST)}"
    if any(t in GATED for t in toks) and not approved_by.strip():
        return [], f"{sorted(set(toks) & GATED)} require a non-empty approvedBy"
    return toks, None


def _run_command(root: Path, cmd: dict) -> dict:
    cid = cmd.get("id", "")
    args_str = str(cmd.get("args", ""))
    approved_by = str(cmd.get("approvedBy", ""))
    toks, reason = _validate(args_str, approved_by)
    started = _now()
    if reason:
        return {"id": cid, "args": args_str, "status": "rejected", "reason": reason,
                "startedAt": started, "endedAt": _now(), "exitCode": None,
                "stdoutTail": "", "artifactsTouched": []}
    runner = root / RUNNER
    if not runner.exists():
        return {"id": cid, "args": args_str, "status": "error",
                "reason": f"{RUNNER} not found", "startedAt": started,
                "endedAt": _now(), "exitCode": None, "stdoutTail": "",
                "artifactsTouched": []}
    before = {a["path"]: a["mtime"] for a in _artifact_index(root)}
    proc = subprocess.run(["bash", RUNNER, *toks], cwd=str(root),
                          capture_output=True, text=True)
    after = _artifact_index(root)
    touched = [a["path"] for a in after if before.get(a["path"]) != a["mtime"]]
    tail = (proc.stdout or "")[-MAX_TAIL:]
    return {"id": cid, "args": args_str, "status": "ok" if proc.returncode == 0 else "error",
            "exitCode": proc.returncode, "startedAt": started, "endedAt": _now(),
            "approvedBy": approved_by, "stdoutTail": tail,
            "stderrTail": (proc.stderr or "")[-4096:], "artifactsTouched": touched}


def tick(root: Path, branch: str, trainwatch_url: str) -> None:
    _git("fetch", "origin", branch, cwd=root, check=False)
    _git("checkout", branch, cwd=root, check=False)
    _git("reset", "--hard", f"origin/{branch}", cwd=root, check=False)

    bridge = root / "bridge"
    (bridge / "commands").mkdir(parents=True, exist_ok=True)
    (bridge / "results").mkdir(parents=True, exist_ok=True)

    cmds = sorted((bridge / "commands").glob("*.json"))
    done = {p.stem for p in (bridge / "results").glob("*.json")}
    pending = [p for p in cmds if p.stem not in done]

    # publish status
    status = {
        "updatedAt": _now(), "host": "spark-2f2d",
        "trainwatch": _trainwatch_snapshot(trainwatch_url),
        "artifacts": _artifact_index(root),
        "pendingCommands": [p.stem for p in pending],
    }
    (bridge / "STATUS.json").write_text(json.dumps(status, indent=2) + "\n")

    ran = []
    for cpath in pending:
        try:
            cmd = json.loads(cpath.read_text())
        except Exception as exc:  # noqa: BLE001
            res = {"id": cpath.stem, "status": "rejected",
                   "reason": f"unparseable command: {exc}", "endedAt": _now()}
        else:
            sys.stderr.write(f"[bridge] running {cpath.stem}: {cmd.get('args')!r}\n")
            res = _run_command(root, cmd)
        (bridge / "results" / f"{res['id'] or cpath.stem}.json").write_text(
            json.dumps(res, indent=2) + "\n")
        ran.append(res.get("id") or cpath.stem)

    _git("add", "bridge", cwd=root, check=False)
    status_line = f"bridge: status @ {status['updatedAt']}"
    if ran:
        status_line += f" + {len(ran)} result(s): {', '.join(ran)}"
    commit = _git("commit", "-m", status_line, cwd=root, check=False)
    if commit.returncode == 0:
        _push_with_retry(root, branch)
    # else: nothing changed (no new status diff) -> skip push


def main() -> int:
    ap = argparse.ArgumentParser(description="GitHub-mediated Spark bridge poller")
    ap.add_argument("--repo-dir", default=".", type=Path)
    ap.add_argument("--branch", default="spark-bridge")
    ap.add_argument("--trainwatch", default="http://127.0.0.1:8420/api/runs")
    ap.add_argument("--interval", type=int, default=30, help="seconds between ticks")
    ap.add_argument("--once", action="store_true", help="single tick then exit")
    a = ap.parse_args()
    root = a.repo_dir.resolve()
    sys.stderr.write(f"[bridge] poller on {root} branch={a.branch} "
                     f"interval={a.interval}s once={a.once}\n")
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

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Secure local web control panel for the sophia-agi DGX-Spark cluster.

design/infra; no capability claim; canClaimAGI stays false.

A button-row web UI that triggers ONLY allowlisted, fixed-argv commands and streams their stdout
line-by-line to a live console via Server-Sent Events. A web button that runs commands is an RCE
surface, so the security posture is deliberately narrow:

  * ALLOWLIST-ONLY. An action is ``{id, label, argv (a FIXED list), gpu}``. There is NO free-form
    command field, ever. ``subprocess.Popen(argv, shell=False)`` — never ``shell=True``, never any
    user-interpolated token. The server refuses any action id not in ``ACTION_REGISTRY``.
  * LOCALHOST BY DEFAULT. Binds ``127.0.0.1``. ``--host`` may override, but a non-localhost host
    REQUIRES ``--token`` (the server refuses to start without one), and then every ``/run`` and
    ``/stream`` must present it (constant-time compare). Reads (``/status``) may be tokenless only
    on localhost.
  * ONE-GPU INVARIANT. GPU actions (``cert-t1`` / ``bench-a``) check ``spark_bridge.gpu_is_free``
    via ``read_status`` BEFORE launching and refuse if a job is running. v1 runs the fixed argv
    directly (with the gpu-free guard); ``--bridge-dir`` enables the preferred path of composing the
    command through ``spark_bridge.build_command`` and writing it to ``bridge/commands/`` so the
    Spark poller — which enforces the human-approval gate — executes it. See the doc for the choice.

The action-registry + auth + argv logic is PURE and unit-tested (``--selftest`` /
``tests/test_spark_control_panel.py``); the HTTP server + subprocess streaming is the impure part
exercised live, not in CI. The human who presses the button IS the approver (the no-AI-self-approval
rule, ``docs/11-Platform/Spark-Bridge-Cloud-Operator.md``).

Run (on the Spark, localhost):
    python tools/spark_control_panel.py
    # open http://localhost:8765   (or over SSH: ssh -L 8765:localhost:8765 spark)

Never expose this publicly without Cloudflare Access + a token — see docs/11-Platform/Spark-Control-Panel.md.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# spark_bridge is reused for the gpu-free check and the (optional) bridge dispatch path.
try:
    from tools import spark_bridge  # type: ignore
except Exception:  # noqa: BLE001 - allow running as a loose script
    import spark_bridge  # type: ignore


# --- ACTION REGISTRY (editable) ----------------------------------------------------------------
# An action's argv is a FIXED list. No element is ever built from request data. ``gpu`` actions are
# gated behind the one-GPU-free check (and, off-localhost, the token). To add an action, append a
# fixed-argv entry here — there is intentionally no other way to introduce a command.
_PY = sys.executable or "python3"

ACTION_REGISTRY: "dict[str, dict]" = {
    "bridge-status": {
        "label": "Bridge status",
        "argv": [_PY, "tools/spark_bridge.py", "status"],
        "gpu": False,
    },
    "trainwatch": {
        "label": "TrainWatch (live ETA)",
        "argv": [_PY, "tools/spark_bridge.py", "trainwatch"],
        "gpu": False,
    },
    "gpu-free": {
        # Served specially (reads spark_bridge.read_status in-process, no subprocess) but still a
        # first-class registry action so it shows as a button and goes through the same auth.
        "label": "GPU free?",
        "argv": None,
        "gpu": False,
    },
    "link-results": {
        "label": "Link results -> TrainWatch",
        "argv": [_PY, "tools/trainwatch_link_results.py", "--dry-run",
                 "--glob", "agi-proof/benchmark-results/*.json"],
        "gpu": False,
    },
    "board-refresh": {
        "label": "Benchmark board refresh",
        "argv": [_PY, "tools/trainwatch_benchmark_board.py", "--queue", "--dry-run",
                 "--glob", "agi-proof/benchmark-results/*.json"],
        "gpu": False,
    },
    # --- GPU actions: gated behind gpu-free + (off-localhost) the token ---
    "cert-t1": {
        "label": "Certify T1 (GPU)",
        "argv": ["bash", "scripts/run_local_benchmarks.sh", "--all"],
        "gpu": True,
        # bridge dispatch composes this when --bridge-dir is given (preferred path).
        "bridge_args": "--all --execute",
    },
    "bench-a": {
        "label": "Bench A (GPU)",
        "argv": ["bash", "scripts/run_local_benchmarks.sh", "--bench-a"],
        "gpu": True,
        "bridge_args": "--bench-a --execute",
    },
}

# Shell control characters that must NEVER appear in a fixed argv token (defense-in-depth: argv is
# already passed without a shell, so these are inert here — but we assert their absence so a
# careless registry edit that someone later pipes through a shell can't smuggle an injection).
# NOTE: glob chars (* ? [ ]) are intentionally NOT here — argv tokens never reach a shell, so a
# token like 'agi-proof/benchmark-results/*.json' is a literal pattern the *tool* itself globs.
_SHELL_META = set(";|&$`<>\n\r\\\"'")

MAX_GPU_JOBS = 1
MAX_TOTAL_JOBS = 4


# --- PURE helpers (unit-tested) ----------------------------------------------------------------
def resolve_action(action_id: str) -> dict:
    """Return the registry entry for ``action_id`` or raise ``KeyError``. The single chokepoint:
    every code path that runs anything goes through here, so an unknown / free-form / injected id
    can never reach a subprocess."""
    if not isinstance(action_id, str) or action_id not in ACTION_REGISTRY:
        raise KeyError(f"unknown action id: {action_id!r}")
    return ACTION_REGISTRY[action_id]


def is_gpu_action(action_id: str) -> bool:
    """True iff the action is GPU-bound (must pass the gpu-free gate before launch)."""
    return bool(resolve_action(action_id).get("gpu"))


_LOCALHOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def is_localhost(host: str) -> bool:
    """Loopback hosts that may serve tokenless READS (and don't require a token to start).
    Note ``0.0.0.0`` is NOT loopback (it binds all interfaces) so it requires a token."""
    return host in _LOCALHOSTS


def auth_ok(host: str, token: "str | None", provided: "str | None", *, is_read: bool = False) -> bool:
    """Constant-time auth decision.

    Rules:
      * On localhost a READ (``is_read=True``) is allowed with no token.
      * Otherwise a token MUST be configured AND the request MUST present a matching one
        (``hmac.compare_digest`` constant-time compare).
    A configured token is always enforced even on localhost for writes — once you set a token, it is
    required for /run and /stream regardless of host.
    """
    if is_read and is_localhost(host) and not token:
        return True
    if not token:
        # No token configured: permitted only for localhost (reads handled above, writes here).
        return is_localhost(host)
    if not provided:
        return False
    return hmac.compare_digest(str(token), str(provided))


def host_requires_token(host: str) -> bool:
    """A non-localhost bind requires a token; the server refuses to start otherwise."""
    return not is_localhost(host)


def argv_is_safe(argv: "list | None") -> bool:
    """True iff argv is None (in-process action) or a list of str tokens with no shell metachars.
    Asserts the registry invariant; not a runtime defense (we never use a shell) but a guard against
    a careless registry edit."""
    if argv is None:
        return True
    if not isinstance(argv, list) or not argv:
        return False
    for tok in argv:
        if not isinstance(tok, str) or not tok:
            return False
        if any(c in _SHELL_META for c in tok):
            return False
    return True


def public_actions() -> list:
    """The /actions payload: {id,label,gpu} only (never argv)."""
    return [{"id": k, "label": v["label"], "gpu": bool(v.get("gpu"))}
            for k, v in ACTION_REGISTRY.items()]


def gpu_free_now() -> "tuple[bool, dict | None]":
    """Read live bridge status and decide if the GPU is free (the one-GPU invariant). Returns
    (free, status). Impure (does a git read via spark_bridge); kept out of unit tests."""
    status = spark_bridge.read_status()
    if status is None:
        # No status available -> treat as NOT free (fail closed for GPU launches).
        return False, None
    return spark_bridge.gpu_is_free(status), status


# --- impure: job table + subprocess streaming --------------------------------------------------
class Job:
    __slots__ = ("id", "action_id", "gpu", "proc", "lines", "done", "lock", "cv", "rc")

    def __init__(self, action_id: str, gpu: bool):
        self.id = uuid.uuid4().hex
        self.action_id = action_id
        self.gpu = gpu
        self.proc: "subprocess.Popen | None" = None
        self.lines: list[str] = []
        self.done = False
        self.rc: "int | None" = None
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)


class JobManager:
    def __init__(self) -> None:
        self.jobs: "dict[str, Job]" = {}
        self.lock = threading.Lock()

    def _counts(self) -> "tuple[int, int]":
        active = [j for j in self.jobs.values() if not j.done]
        return len(active), len([j for j in active if j.gpu])

    def can_start(self, gpu: bool) -> "tuple[bool, str]":
        with self.lock:
            total, gpus = self._counts()
            if total >= MAX_TOTAL_JOBS:
                return False, f"too many concurrent jobs ({total}/{MAX_TOTAL_JOBS})"
            if gpu and gpus >= MAX_GPU_JOBS:
                return False, "a GPU job is already running (one-GPU invariant)"
            return True, ""

    def start(self, action_id: str) -> Job:
        entry = resolve_action(action_id)
        argv = entry["argv"]
        job = Job(action_id, bool(entry.get("gpu")))
        with self.lock:
            self.jobs[job.id] = job
        # argv built ONLY from the fixed registry entry; cwd pinned to ROOT; no shell.
        proc = subprocess.Popen(
            argv, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        job.proc = proc
        threading.Thread(target=self._pump, args=(job,), daemon=True).start()
        return job

    def start_inprocess(self, action_id: str, producer) -> Job:
        """For actions whose work is an in-process call (e.g. gpu-free reads status)."""
        entry = resolve_action(action_id)
        job = Job(action_id, bool(entry.get("gpu")))
        with self.lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run_producer, args=(job, producer), daemon=True).start()
        return job

    def _emit(self, job: Job, line: str) -> None:
        with job.cv:
            job.lines.append(line)
            job.cv.notify_all()

    def _finish(self, job: Job, rc: int) -> None:
        with job.cv:
            job.rc = rc
            job.done = True
            job.cv.notify_all()

    def _pump(self, job: Job) -> None:
        try:
            assert job.proc is not None and job.proc.stdout is not None
            for line in job.proc.stdout:
                self._emit(job, line.rstrip("\n"))
            rc = job.proc.wait()
        except Exception as exc:  # noqa: BLE001
            self._emit(job, f"[panel] error: {exc}")
            rc = 1
        self._finish(job, rc)

    def _run_producer(self, job: Job, producer) -> None:
        rc = 0
        try:
            for line in producer():
                self._emit(job, line)
        except Exception as exc:  # noqa: BLE001
            self._emit(job, f"[panel] error: {exc}")
            rc = 1
        self._finish(job, rc)

    def stream(self, job_id: str):
        """Generator of (kind, payload) tuples: ('line', text) then ('done', rc). Blocks for new
        lines via the per-job condition variable. ``job_id`` is server-generated; never a path."""
        job = self.jobs.get(job_id)
        if job is None:
            yield ("done", -1)
            return
        idx = 0
        while True:
            with job.cv:
                while idx >= len(job.lines) and not job.done:
                    job.cv.wait(timeout=30)
                while idx < len(job.lines):
                    yield ("line", job.lines[idx])
                    idx += 1
                if job.done and idx >= len(job.lines):
                    yield ("done", job.rc if job.rc is not None else 0)
                    return


# --- HTML (self-contained, no external deps) ---------------------------------------------------
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sophia-agi Spark control panel</title>
<style>
  :root{color-scheme:dark}
  body{font:14px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0;background:#0d1117;color:#c9d1d9}
  header{padding:14px 18px;border-bottom:1px solid #21262d;background:#161b22}
  h1{font-size:15px;margin:0;font-weight:600}
  .sub{color:#8b949e;font-size:12px;margin-top:4px}
  #bar{display:flex;flex-wrap:wrap;gap:8px;padding:14px 18px}
  button{font:inherit;cursor:pointer;border:1px solid #30363d;background:#21262d;color:#c9d1d9;
         padding:8px 12px;border-radius:6px}
  button:hover{background:#30363d}
  button.gpu{border-color:#9e6a03;color:#f0c674}
  button:disabled{opacity:.5;cursor:not-allowed}
  #log{margin:0 18px 18px;padding:12px;background:#010409;border:1px solid #21262d;border-radius:6px;
       height:60vh;overflow:auto;white-space:pre-wrap;word-break:break-word}
  .meta{color:#8b949e}
  .gpu-line{color:#f0c674}
  .done{color:#3fb950}
  .fail{color:#f85149}
</style></head>
<body>
<header>
  <h1>sophia-agi &mdash; Spark control panel</h1>
  <div class="sub">Allowlisted fixed-argv actions only. No free-form commands. design/infra; canClaimAGI stays false.</div>
</header>
<div id="bar"></div>
<pre id="log"><span class="meta">Pick an action above. Output streams here.</span>\n</pre>
<script>
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const logEl = document.getElementById("log");
const bar = document.getElementById("bar");
let es = null;
function authHeaders(){ return TOKEN ? {"X-Panel-Token": TOKEN} : {}; }
function log(text, cls){
  const span = document.createElement("span");
  if(cls) span.className = cls;
  span.textContent = text + "\\n";
  logEl.appendChild(span);
  logEl.scrollTop = logEl.scrollHeight;
}
function setBusy(b){ for(const el of bar.querySelectorAll("button")) el.disabled = b; }
async function loadActions(){
  const r = await fetch("/actions");
  const actions = await r.json();
  for(const a of actions){
    const btn = document.createElement("button");
    btn.textContent = a.label + (a.gpu ? " \\u26a1" : "");
    if(a.gpu) btn.className = "gpu";
    btn.onclick = () => run(a.id, a.label, a.gpu);
    bar.appendChild(btn);
  }
}
async function run(id, label, gpu){
  if(es){ es.close(); es = null; }
  setBusy(true);
  log("\\n$ " + label + (gpu ? "  (GPU)" : ""), gpu ? "gpu-line" : "meta");
  try{
    const r = await fetch("/run", {method:"POST",
      headers: Object.assign({"Content-Type":"application/json"}, authHeaders()),
      body: JSON.stringify({action: id})});
    const data = await r.json();
    if(!r.ok || data.error){ log("REFUSED: " + (data.error||r.status), "fail"); setBusy(false); return; }
    openStream(data.jobId);
  }catch(e){ log("error: " + e, "fail"); setBusy(false); }
}
function openStream(jobId){
  const url = "/stream?job=" + encodeURIComponent(jobId) + (TOKEN ? "&token=" + encodeURIComponent(TOKEN) : "");
  es = new EventSource(url);
  es.onmessage = (ev) => log(ev.data);
  es.addEventListener("done", (ev) => {
    const rc = (ev.data||"").trim();
    log("[done] exit " + rc, rc === "0" ? "done" : "fail");
    es.close(); es = null; setBusy(false);
  });
  es.onerror = () => { log("[stream closed]", "meta"); if(es){es.close(); es=null;} setBusy(false); };
}
loadActions();
</script>
</body></html>
"""


# --- HTTP handler ------------------------------------------------------------------------------
def make_handler(host: str, token: "str | None", manager: JobManager, bridge_dir: "Path | None"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SparkPanel/1.0"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter, but keep a one-line audit trail
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        # -- helpers --
        def _provided_token(self, qs: dict) -> "str | None":
            return self.headers.get("X-Panel-Token") or (qs.get("token", [None])[0])

        def _send_json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, body: str) -> None:
            data = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        # -- routes --
        def do_GET(self):
            parsed = urlparse(self.path)
            path, qs = parsed.path, parse_qs(parsed.query)
            if path == "/":
                self._send_html(_PAGE)
                return
            if path == "/actions":
                body = json.dumps(public_actions()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/status":
                if not auth_ok(host, token, self._provided_token(qs), is_read=True):
                    self._send_json(401, {"error": "token required"})
                    return
                free, status = gpu_free_now()
                self._send_json(200, {"gpuFree": free, "status": status})
                return
            if path == "/stream":
                self._handle_stream(qs)
                return
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/run":
                self._send_json(404, {"error": "not found"})
                return
            qs = parse_qs(parsed.query)
            if not auth_ok(host, token, self._provided_token(qs), is_read=False):
                self._send_json(401, {"error": "token required"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw or b"{}")
                action_id = payload.get("action")
            except Exception:  # noqa: BLE001
                self._send_json(400, {"error": "bad json"})
                return
            # The chokepoint: unknown / free-form / injected ids are refused here.
            try:
                entry = resolve_action(action_id)
            except KeyError:
                self._send_json(400, {"error": f"unknown action: {action_id!r}"})
                return
            gpu = bool(entry.get("gpu"))
            if gpu:
                free, _status = gpu_free_now()
                if not free:
                    self._send_json(409, {"error": "GPU busy (one-GPU invariant) — refused"})
                    return
            ok, why = manager.can_start(gpu)
            if not ok:
                self._send_json(409, {"error": why})
                return
            # GPU job + bridge dispatch: compose a command JSON into bridge/commands/ (preferred).
            if gpu and bridge_dir is not None:
                try:
                    job = self._dispatch_via_bridge(action_id, entry)
                except Exception as exc:  # noqa: BLE001
                    self._send_json(500, {"error": f"bridge dispatch failed: {exc}"})
                    return
                self._send_json(200, {"jobId": job.id, "dispatched": "bridge"})
                return
            if entry["argv"] is None:
                job = manager.start_inprocess(action_id, _gpu_free_producer)
            else:
                job = manager.start(action_id)
            self._send_json(200, {"jobId": job.id})

        def _dispatch_via_bridge(self, action_id: str, entry: dict) -> Job:
            """Compose + write a bridge command JSON (no GPU run from here). The Spark poller, with
            the human-approval gate, is what actually executes it. approvedBy carries the
            button-press as the human approval (the panel's human IS the approver)."""
            cmd_id = f"panel-{action_id}-{uuid.uuid4().hex[:8]}"
            cmd = spark_bridge.build_command(
                cmd_id, entry["bridge_args"], created_by="spark_control_panel",
                approved_by=f"panel button press: {action_id} ({host})",
                note="dispatched from the local Spark control panel",
            )
            cmds = bridge_dir / "bridge" / "commands"
            cmds.mkdir(parents=True, exist_ok=True)
            out = cmds / f"{cmd_id}.json"
            out.write_text(json.dumps(cmd, indent=2) + "\n")

            def producer():
                yield f"[panel] composed bridge command -> {out}"
                yield f"[panel] id={cmd_id} args={entry['bridge_args']!r}"
                yield "[panel] commit/push bridge/commands and the Spark poller will execute it"
                yield "complete (exit 0)"
            return manager.start_inprocess(action_id, producer)

        def _handle_stream(self, qs: dict):
            if not auth_ok(host, token, self._provided_token(qs), is_read=False):
                self._send_json(401, {"error": "token required"})
                return
            job_id = qs.get("job", [None])[0]
            if not job_id or job_id not in manager.jobs:
                self._send_json(404, {"error": "no such job"})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            try:
                for kind, payload in manager.stream(job_id):
                    if kind == "line":
                        chunk = f"data: {payload}\n\n"
                    else:
                        chunk = f"event: done\ndata: {payload}\n\n"
                    self.wfile.write(chunk.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


def _gpu_free_producer():
    free, status = gpu_free_now()
    yield f"[panel] gpu_is_free = {free}"
    if status is None:
        yield "[panel] no STATUS.json (bridge unreachable / not fetched)"
    else:
        yield f"[panel] running = {status.get('running')!r}  pending = {status.get('pendingCommands')!r}"
    yield "complete (exit 0)"


# --- selftest (pure) ---------------------------------------------------------------------------
def _selftest() -> int:
    checks: "dict[str, bool]" = {}

    # registry integrity: every argv is safe; gpu actions carry bridge_args.
    checks["registry_argv_safe"] = all(argv_is_safe(v["argv"]) for v in ACTION_REGISTRY.values())
    checks["gpu_actions_have_bridge_args"] = all(
        v.get("bridge_args") for v in ACTION_REGISTRY.values() if v.get("gpu"))

    # resolve_action refuses unknown / injection ids.
    refused = 0
    for bad in ("nope", "; rm -rf /", "bridge-status; ls", "", None, "../../etc/passwd"):
        try:
            resolve_action(bad)  # type: ignore[arg-type]
        except KeyError:
            refused += 1
    checks["unknown_ids_refused"] = refused == 6
    checks["known_id_resolves"] = resolve_action("bridge-status")["label"] == "Bridge status"

    # is_gpu_action correctness.
    checks["is_gpu_action"] = (
        is_gpu_action("cert-t1") and is_gpu_action("bench-a")
        and not is_gpu_action("bridge-status") and not is_gpu_action("gpu-free"))

    # auth: localhost read exempt; non-localhost requires a matching token.
    checks["localhost_read_exempt"] = auth_ok("127.0.0.1", None, None, is_read=True)
    checks["localhost_write_ok_no_token"] = auth_ok("127.0.0.1", None, None, is_read=False)
    checks["remote_no_token_refused"] = not auth_ok("10.0.0.5", None, None, is_read=False)
    checks["remote_no_token_read_refused"] = not auth_ok("10.0.0.5", None, None, is_read=True)
    checks["remote_token_required_present"] = auth_ok("10.0.0.5", "sek", "sek", is_read=False)
    checks["remote_token_mismatch_refused"] = not auth_ok("10.0.0.5", "sek", "wrong", is_read=False)
    checks["configured_token_enforced_localhost"] = (
        auth_ok("127.0.0.1", "sek", "sek", is_read=False)
        and not auth_ok("127.0.0.1", "sek", "wrong", is_read=False))

    # host_requires_token.
    checks["host_requires_token"] = (
        host_requires_token("0.0.0.0") and host_requires_token("10.0.0.5")
        and not host_requires_token("127.0.0.1") and not host_requires_token("localhost"))

    # argv_is_safe rejects shell metacharacters.
    checks["argv_safe_rejects_meta"] = (
        not argv_is_safe(["bash", "-c", "rm -rf /; echo $HOME"])
        and not argv_is_safe(["python", "x | y"])
        and argv_is_safe([_PY, "tools/spark_bridge.py", "status"]))

    # public_actions never leaks argv.
    checks["public_actions_no_argv"] = all(
        set(a.keys()) == {"id", "label", "gpu"} for a in public_actions())

    ok = all(checks.values())
    print("spark_control_panel selftest:", "PASS" if ok else "FAIL")
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    return 0 if ok else 1


# --- CLI ---------------------------------------------------------------------------------------
def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind host (default 127.0.0.1; non-localhost REQUIRES --token)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=os.environ.get("PANEL_TOKEN") or None,
                    help="shared secret; required when --host is not localhost")
    ap.add_argument("--bridge-dir", default=None,
                    help="path to a bridge checkout; GPU actions then DISPATCH via bridge/commands/ "
                         "instead of running directly")
    ap.add_argument("--selftest", action="store_true", help="run the pure-logic selftest and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    # SECURITY: refuse to start a non-localhost bind without a token.
    if host_requires_token(args.host) and not args.token:
        print(f"REFUSED: --host {args.host} is not localhost; a --token is REQUIRED to bind it "
              f"(a panel that runs commands must never be tokenless off-loopback).", file=sys.stderr)
        return 2

    bridge_dir = Path(args.bridge_dir).resolve() if args.bridge_dir else None
    if bridge_dir is not None and not bridge_dir.is_dir():
        print(f"REFUSED: --bridge-dir {bridge_dir} is not a directory", file=sys.stderr)
        return 2

    manager = JobManager()
    handler = make_handler(args.host, args.token, manager, bridge_dir)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)

    where = f"http://{args.host}:{args.port}"
    print(f"[spark_control_panel] serving {where}")
    print(f"[spark_control_panel] host={args.host} token={'set' if args.token else 'none'} "
          f"bridge_dir={bridge_dir or 'none'}")
    if args.token:
        print(f"[spark_control_panel] open {where}/?token=<token>  (UI passes it as X-Panel-Token)")
    print("[spark_control_panel] allowlisted actions:",
          ", ".join(ACTION_REGISTRY.keys()))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[spark_control_panel] shutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

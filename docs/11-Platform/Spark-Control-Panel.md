# Spark Control Panel

> design/infra; no capability claim; canClaimAGI stays false.

A small, stdlib-only local web control panel for the sophia-agi DGX-Spark cluster: a row of buttons
that trigger **allowlisted, fixed-argv** commands, with a **live console** that streams each command's
stdout line-by-line over Server-Sent Events. Built on `http.server` — no Flask, no external deps.

Tool: `tools/spark_control_panel.py` · Tests: `tests/test_spark_control_panel.py`
Pairs with `tools/spark_bridge.py` (reused for the GPU-free check and the optional bridge dispatch)
and `docs/11-Platform/Spark-Bridge-Cloud-Operator.md` (the no-AI-self-approval + one-GPU rules).

---

## What it is

A web button that runs commands is an RCE surface, so the design is deliberately narrow. The panel
holds an **ACTION_REGISTRY** of fixed actions; each action is `{id, label, argv (a FIXED list), gpu}`.
There is **no free-form command field, ever**. The server refuses any `action` id that is not in the
registry, and runs the argv with `subprocess.Popen(argv, shell=False)` — never `shell=True`, never a
token interpolated from the request.

The human who presses the button **is the approver** (the no-AI-self-approval rule): a GPU action
only launches when the one-GPU-job invariant says the GPU is free, and a human chose to press it.

### v1 actions

| id              | label                       | gpu | what it runs                                                                         |
|-----------------|-----------------------------|-----|--------------------------------------------------------------------------------------|
| `bridge-status` | Bridge status               | no  | `python tools/spark_bridge.py status`                                                |
| `trainwatch`    | TrainWatch (live ETA)       | no  | `python tools/spark_bridge.py trainwatch`                                             |
| `gpu-free`      | GPU free?                   | no  | in-process `spark_bridge.read_status` -> `gpu_is_free`                                |
| `link-results`  | Link results -> TrainWatch  | no  | `python tools/trainwatch_link_results.py --dry-run --glob 'agi-proof/benchmark-results/*.json'` |
| `board-refresh` | Benchmark board refresh     | no  | `python tools/trainwatch_benchmark_board.py --queue --dry-run --glob 'agi-proof/benchmark-results/*.json'` |
| `cert-t1`       | Certify T1 (GPU)            | yes | `bash scripts/run_local_benchmarks.sh --all` (gated behind GPU-free + token)          |
| `bench-a`       | Bench A (GPU)               | yes | `bash scripts/run_local_benchmarks.sh --bench-a` (gated behind GPU-free + token)      |

To add an action, append a fixed-argv entry to `ACTION_REGISTRY`. There is intentionally no other way
to introduce a command.

---

## Run it on the Spark (localhost)

```bash
python tools/spark_control_panel.py          # binds 127.0.0.1:8765 by default
# then open http://localhost:8765
```

From your laptop, tunnel over SSH (no public exposure):

```bash
ssh -L 8765:localhost:8765 spark
# now http://localhost:8765 on your laptop reaches the panel on the Spark
```

CLI flags:

- `--host` (default `127.0.0.1`) — non-localhost binds **require** `--token` (refuses to start otherwise).
- `--port` (default `8765`).
- `--token` (or `PANEL_TOKEN` env) — shared secret; required off-localhost; enforced for `/run` and
  `/stream` everywhere once set. Pass it in the URL as `?token=...` (the UI forwards it as the
  `X-Panel-Token` header).
- `--bridge-dir PATH` — a bridge checkout; with it set, GPU actions **dispatch through the bridge**
  (compose a command JSON into `bridge/commands/`) instead of running the argv directly. See below.
- `--selftest` — run the pure-logic checks and exit.

### Endpoints

- `GET /` — the single self-contained HTML page (inline CSS/JS, no external requests).
- `GET /actions` — JSON `[{id,label,gpu}]` (never argv).
- `GET /status` — `{gpuFree, status}` (tokenless only on localhost).
- `POST /run {action}` — validate id ∈ registry + auth + (GPU) GPU-free, spawn the job, return `{jobId}`.
- `GET /stream?job=ID` — `text/event-stream`; one `data:` event per stdout line, then `event: done`.
  The `job` id is **server-generated** (a uuid), never a client-supplied path.

---

## Security model (the #1 requirement)

**Threat:** anyone who can reach the panel can press a button that runs a command on the Spark. So the
panel must never (a) accept an arbitrary command, (b) be reachable by anyone but the owner.

Controls:

1. **Allowlist-only, no shell.** Only the seven fixed `ACTION_REGISTRY` ids are accepted; `resolve_action`
   is the single chokepoint every run path goes through. argv is a fixed list run with `shell=False`;
   no request data is ever interpolated into a command. Unit tests assert that classic injection shapes
   (`; rm -rf /`, `$(reboot)`, `` `id` ``, `../../etc/passwd`, …) are refused, and that no registry
   argv contains a shell control character.
2. **Localhost by default.** Binds `127.0.0.1`. A non-localhost `--host` requires `--token` or the
   server refuses to start. Reads (`/status`) are tokenless **only** on localhost; `/run` and `/stream`
   require the token whenever one is configured, compared in constant time (`hmac.compare_digest`).
3. **No path traversal / no query-driven file reads.** The only client-supplied identifiers are the
   action id (allowlist-checked) and the SSE job id (a server-minted uuid looked up in an in-memory
   table). No request value becomes a filesystem path.
4. **One-GPU invariant.** A GPU action checks `spark_bridge.gpu_is_free(read_status())` immediately
   before launch and refuses (HTTP 409) if a job is running or pending — and the manager caps
   concurrency at 1 GPU + a few reads. If `STATUS.json` is unreachable, GPU launches **fail closed**.

### Direct run vs. bridge dispatch (the v1 choice)

For v1, a GPU button runs the fixed argv **directly** on the Spark, guarded by the GPU-free check.
This is acceptable because the argv is fixed and the presser is the human approver. When `--bridge-dir`
is given, the **preferred** path is taken instead: the panel composes the command via
`spark_bridge.build_command(...)` (which enforces the allowlist and the human-`approvedBy` gate) and
writes it to `bridge/commands/<id>.json`; committing/pushing that lets the Spark poller execute it
under the same discipline the cloud operator uses. Run it that way when you want the bridge's audit
trail and serializer; run direct for a quick local-only v1.

---

## Cloudflare path — expose at `panel.<domain>` securely

To reach the panel from your Cloudflare domain, use a **Cloudflare Tunnel** (`cloudflared`) so nothing
is opened on the Spark's public IP, and put it **behind Cloudflare Access (Zero Trust)** so only your
identity can load it.

> **WARNING — never expose this panel publicly without BOTH Cloudflare Access AND a `--token`.**
> It runs commands on the Spark. Access is the front door (only the owner's email gets in); the token
> is defense-in-depth if Access is ever misconfigured. Without them, a stray request is an RCE.

### 1. Run the panel with a token (still bound to localhost)

The tunnel connects to it over loopback, so keep the bind on `127.0.0.1` and set a token for `/run`:

```bash
PANEL_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  python tools/spark_control_panel.py --host 127.0.0.1 --port 8765 --token "$PANEL_TOKEN"
# open it at https://panel.<domain>/?token=<PANEL_TOKEN>
```

### 2. `cloudflared` tunnel config — `~/.cloudflared/config.yml`

```yaml
tunnel: spark-panel                       # the named tunnel (cloudflared tunnel create spark-panel)
credentials-file: /home/USER/.cloudflared/<TUNNEL-UUID>.json

ingress:
  - hostname: panel.example.com           # your domain
    service: http://localhost:8765        # the panel, on loopback
  - service: http_status:404              # default: refuse everything else
```

```bash
cloudflared tunnel create spark-panel
cloudflared tunnel route dns spark-panel panel.example.com
cloudflared tunnel run spark-panel        # (or install as a service)
```

### 3. Cloudflare Access policy (Zero Trust) — REQUIRED

In the Cloudflare Zero Trust dashboard, add an **Access > Application** (self-hosted) for
`panel.example.com` with a single **Allow** policy:

- **Include:** Emails -> *only the owner's email address* (your account email).
- Everything else is denied by default.

This forces a Cloudflare login before any request reaches the tunnel, so the panel is never publicly
reachable. The `--token` then gates `/run` / `/stream` as a second factor.

> Optional hardening: a Cloudflare **Service Token** for non-interactive access, and an Access policy
> that also requires a device posture / country rule. The minimum bar is: Access allow-list = owner
> only, **plus** the panel token.

---

## Honest limits

- This is a **control plane** for owned-hardware (Spark) benchmarks. Real metered GPU (RunPod) still
  goes through the repo's GitHub Actions guardrail, not this panel.
- The panel does not see live cluster ownership (`SESSION-COORDINATION.md` is untracked); the GPU-free
  check reads the bridge `STATUS.json`, and the human pressing the button is responsible for the
  one-job invariant alongside any peer sessions. `canClaimAGI` stays false.

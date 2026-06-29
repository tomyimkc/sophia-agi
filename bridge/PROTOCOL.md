# Spark <-> Cloud-Agent GitHub Bridge

A git-backed message queue on the `spark-bridge` branch. It exists because the
cloud Claude Code session can only reach **allowlisted** hosts (GitHub yes, the
Spark's Tailscale Funnel no — egress policy returns `connect_rejected 502`). So
both sides talk **only to GitHub**:

- **Spark (Hermes)** runs `tools/github_bridge_poll.py` on a loop: publishes
  status, executes approved commands, writes results, pushes.
- **Cloud agent (Claude)** reads `bridge/STATUS.json` + `bridge/results/*.json`
  via the GitHub MCP, and writes `bridge/commands/*.json` to request work.

This is a **control plane only**. Real RunPod/GPU jobs still go through GitHub
Actions per the repo guardrail; this bridge runs *owned-hardware* Spark/Mac
benchmarks via `scripts/run_local_benchmarks.sh` and nothing else.

## Layout (all on branch `spark-bridge`)
```
bridge/
  STATUS.json          # Spark -> cloud: latest trainwatch snapshot + artifact index
  commands/<id>.json   # cloud -> Spark: a request to run the benchmark script
  results/<id>.json    # Spark -> cloud: outcome for the matching command id
```

## Command schema (`bridge/commands/<id>.json`)
```json
{
  "id": "2026-06-29T0500-bench-a-dry",
  "args": "--bench-a --dry-run",
  "createdBy": "claude",
  "createdAt": "2026-06-29T05:00:00Z",
  "approvedBy": ""
}
```
- `args` tokens MUST all be in the allowlist:
  `--dry-run --bench-a --bench-b --all --execute --run-train`.
- Anything else -> the poller writes a `rejected` result, runs nothing.
- `--execute` and `--run-train` are **gated**: the poller runs them ONLY if
  `approvedBy` is a non-empty human handle. Dry-runs need no approval.
- `id` must be filesystem-safe and unique; the result reuses the same `id`.

## Result schema (`bridge/results/<id>.json`)
```json
{
  "id": "...", "args": "...", "status": "ok|rejected|error",
  "exitCode": 0, "startedAt": "...", "endedAt": "...",
  "stdoutTail": "... (capped)", "reason": "(if rejected)",
  "artifactsTouched": ["agi-proof/benchmark-results/..."]
}
```
A command is considered **done** once a `results/<id>.json` exists.

## STATUS schema (`bridge/STATUS.json`)
```json
{
  "updatedAt": "...", "host": "spark-2f2d",
  "trainwatch": { ... raw /api/runs ... },
  "artifacts": [ {"path": "...", "bytes": 123, "mtime": "..."} ],
  "pendingCommands": ["id", ...]
}
```

## Security
- Only `scripts/run_local_benchmarks.sh` is ever executed, only with allowlisted
  flags. No shell metacharacters are passed; `args` is tokenized and each token
  is checked against the allowlist before exec.
- The poller never reads secrets into the repo. Keep `bridge-info.txt` and any
  tokens **out** of git (add to `.gitignore`).
- Stop the bridge anytime: Ctrl-C the poller; nothing else persists.

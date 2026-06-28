---
name: sophia-runpod-connect
description: >
  Reliably connect to RunPod from anywhere — and always have a fallback. Use when
  you need to reach RunPod (list/inspect pods, check whether a pod has stalled,
  restart a stuck pod, or launch a run) and the RUNPOD_API_KEY may or may not be
  present in the current context. Covers both routes: direct REST/MCP when the key
  is exported, and the GitHub-mediated route (dispatch the runpod-connect workflow,
  which holds the key as an Actions secret) when it is not. Invoke when a RunPod
  call returns 401, when a pod looks stalled, or when asked to "connect to RunPod".
metadata:
  short-description: "Always-available RunPod connection (direct key OR GitHub-mediated)"
---

# Sophia · RunPod connect (always-on)

**There is always a way to reach RunPod.** The API key is not guaranteed to be in
your current context — a public repo never ships it, and an interactive agent
session often runs without it (every RunPod call then returns **401**). But the
key always exists as the GitHub repo **Actions secret `RUNPOD_API_KEY`**, so GitHub
can always connect on your behalf. This skill is the decision tree.

## Decision tree — connect to RunPod

```
Need RunPod?
│
├─ Is RUNPOD_API_KEY in the environment?  (echo / `tools/runpod_connect.py --dry-run`)
│   │
│   ├─ YES → DIRECT route
│   │        • RunPod MCP tools (mcp__runpod__*), or
│   │        • python tools/runpod_connect.py --check        # list + flag stalled pods
│   │        • python tools/runpod_connect.py --check --restart-stalled
│   │        • python tools/runpod_connect.py --pod <ID>      # inspect one pod
│   │        • python tools/runpod_connect.py --terminate <ID> --yes  # delete idle pod (saves $)
│   │        • python tools/runpod_connect.py --reap-exited [--yes]    # clean up leaked EXITED pods
│   │        • python tools/runpod_rlvr.py / runpod_train.py # launch a real run
│   │
│   └─ NO  → GITHUB-MEDIATED route (the fallback that always works)
│            The secret lives in GitHub. Trigger the workflow that runs the SAME
│            checker with the secret injected:
│              • Web:  Actions → "runpod-connect" → Run workflow → action=check
│              • CLI:  gh workflow run runpod-connect.yml -f action=check
│              • Recover a stuck pod:  -f action=restart-stalled
│              • Terminate an idle pod (saves $):
│                  gh workflow run runpod-connect.yml \
│                    -f action=terminate -f pod_id=<ID> -f confirm=<ID>
│                (confirm must equal pod_id — a guard against accidental deletes)
│            Read the result from the run log / `runpod-connect-report` artifact.
```

`tools/runpod_connect.py` is the single source of truth for both routes:
- `--dry-run` (offline): tells you which route is available and how to use it.
- `--check`: resolves the key (env), lists pods, flags **stalled** ones, exits `3`
  if any are stalled (so CI/callers notice), `0` if clean.
- `--check --restart-stalled`: stop+start each stalled pod, then exit `0`.
- Missing key → exits `2` with the exact GitHub-dispatch fallback instructions.

## What "stalled" means (honest bound)

The RunPod REST inventory exposes pod **status and shape**, not on-die telemetry.
A pod is flagged **stalled** when `desiredStatus=RUNNING` but it has **no live
runtime/uptime** — it is supposed to be up but the container is not actually
running. A freshly-launched pod legitimately has no runtime for the first minutes,
so a RUNNING pod younger than the boot grace (`BOOT_GRACE_S`, 5 min) is reported as
**booting**, not stalled — never reap or restart a booting pod. Deeper "running but
hung" stalls need an on-node agent
(`agent/cluster/ssh_provider.py` / DCGM over SSH); this skill does not claim to
see those. Fail-closed: unknown ≠ healthy.

## Never do

- **Never** hard-code or commit a RunPod key. `RUNPOD_API_KEY` comes from the
  environment (direct) or the GitHub Actions secret (mediated) — nowhere else.
- **Never** write this skill or any key under `.claude/skills/**` as plaintext:
  that path is git-crypt-encrypted in this **public** repo, and committing without
  git-crypt unlocked would leak it. This skill lives in `skills/portable/` on
  purpose (no secrets in it).
- **Never** treat "unknown" pod state as healthy.

## Leaked / lingering EXITED pods (a real failure mode)

A pod that shows up as **EXITED** but never disappears is a *leak*, not a respawn:
the launcher (`tools/runpod_rlvr.py`) deletes its pod in a `finally`, and the
in-pod watchdog deletes on container exit — but if the orchestrator is killed
(CI cancel / SIGKILL) **and** the watchdog doesn't fire, the pod exits and is
never reaped. It then bills disk indefinitely and "keeps popping up". Reap it:

```
python tools/runpod_connect.py --reap-exited          # preview (exit 3 if any)
python tools/runpod_connect.py --reap-exited --yes     # delete them
# or GitHub-mediated:  gh workflow run runpod-connect.yml -f action=reap-exited
```

The watchdog itself was hardened to derive its pod id from `RUNPOD_POD_ID` *or*
the hostname, and to log (never silently swallow) a failed self-delete.

## Files

- `tools/runpod_connect.py` — resolver + stalled-pod checker/recovery (stdlib only).
- `.github/workflows/runpod-connect.yml` — the GitHub-mediated route (holds the secret).
- `tests/test_runpod_connect.py` — offline tests (key resolution + stall classification).

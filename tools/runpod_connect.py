#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Canonical RunPod connection resolver + stalled-pod checker.

This is the *one* place that answers "how do I connect to RunPod from here, and
is anything stuck?". It exists because the API key is not always present in the
current context (a public repo never ships it; an interactive agent session may
not have it exported), yet the repo still needs a reliable path to reach RunPod.

Two connection routes, by design — and the whole point is that **at least one is
always available**:

1. **Direct** — ``RUNPOD_API_KEY`` is in the environment (local shell, or the
   RunPod MCP server). We hit the REST API straight away.
2. **GitHub-mediated** — the key is NOT here, but it lives as the repo secret
   ``RUNPOD_API_KEY`` (Settings → Secrets and variables → Actions). Dispatch the
   ``runpod-connect`` workflow (``.github/workflows/runpod-connect.yml``); GitHub
   runs *this same script* with the secret injected and reports back. This is the
   fallback this script tells you about when it can't find a key locally.

What "stalled" means here (honest, REST-only bound): the RunPod REST inventory
exposes pod *status and shape*, not on-die telemetry. So a pod is flagged
**stalled** when RunPod's ``desiredStatus`` is ``RUNNING`` but the pod has no live
``runtime`` (no container uptime) — i.e. it is *supposed* to be up but isn't
actually running. Deeper "hung but technically running" stalls need an on-node
agent (see ``agent/cluster/ssh_provider.py``); this script does not claim to see
them.

Offline by default for inspection:

    python3 tools/runpod_connect.py --dry-run          # no network, explains routes
    python3 tools/runpod_connect.py --check            # needs a key; lists + flags pods
    python3 tools/runpod_connect.py --check --json      # machine-readable
    python3 tools/runpod_connect.py --check --restart-stalled  # stop+start stalled pods
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the exact request helper, API base and error type from the live launcher
# so auth, endpoint and error handling stay in one place.
from tools.runpod_rlvr import RunPodError, _api_request  # noqa: E402

DISPATCH_WORKFLOW = "runpod-connect.yml"
GITHUB_FALLBACK_HINT = (
    "No RUNPOD_API_KEY in this context. Use the GitHub-mediated route instead — "
    "the key is stored as the repo Actions secret RUNPOD_API_KEY:\n"
    "  • Web:  Actions → 'runpod-connect' → Run workflow\n"
    f"  • CLI:  gh workflow run {DISPATCH_WORKFLOW} -f action=check\n"
    "GitHub runs this same script with the secret injected and reports back. "
    "To connect directly instead, export RUNPOD_API_KEY (rpa_...) in this shell."
)


def resolve_api_key(explicit: str | None = None,
                    env: "dict[str, str] | None" = None) -> "tuple[str | None, str]":
    """Resolve the RunPod API key. Returns ``(key_or_None, source)``.

    Resolution order: explicit arg → ``RUNPOD_API_KEY`` env. We deliberately do
    NOT read keys from files in the repo — a public repo must never carry one.
    ``source`` is a short tag ('arg', 'env', or 'missing') for honest logging.
    """

    env = os.environ if env is None else env
    if explicit:
        return explicit, "arg"
    key = (env.get("RUNPOD_API_KEY") or "").strip()
    if key:
        return key, "env"
    return None, "missing"


def classify_pod(pod: "dict[str, Any]") -> "dict[str, Any]":
    """Map one RunPod REST pod object onto a small status verdict.

    Verdicts: ``stalled`` (desired RUNNING but no live runtime), ``running``
    (RUNNING with uptime), ``stopped`` (EXITED/TERMINATED), ``unknown``.
    """

    pod_id = str(pod.get("id") or pod.get("podId") or pod.get("name") or "unknown")
    name = str(pod.get("name") or pod_id)
    desired = str(pod.get("desiredStatus") or pod.get("status") or "").upper()
    runtime = pod.get("runtime") or {}
    if not isinstance(runtime, dict):
        runtime = {}
    uptime = runtime.get("uptimeInSeconds")
    has_runtime = bool(runtime) and bool(uptime)

    if desired == "RUNNING" and not has_runtime:
        verdict = "stalled"
        reason = "desiredStatus=RUNNING but no live runtime/uptime (container not actually up)"
    elif desired == "RUNNING":
        verdict = "running"
        reason = f"running; uptime={uptime}s"
    elif desired in {"EXITED", "TERMINATED", "STOPPED"}:
        verdict = "stopped"
        reason = f"desiredStatus={desired}"
    else:
        verdict = "unknown"
        reason = f"desiredStatus={desired or 'unset'}"

    return {"id": pod_id, "name": name, "desiredStatus": desired or None,
            "uptimeInSeconds": uptime, "verdict": verdict, "reason": reason}


def _list_pods(api_key: str) -> "list[dict[str, Any]]":
    pods = _api_request("GET", "/pods", api_key, timeout=60) or []
    if isinstance(pods, dict):  # some API shapes wrap the list
        pods = pods.get("pods") or pods.get("data") or []
    return [p for p in pods if isinstance(p, dict)]


def get_pod(api_key: str, pod_id: str) -> "dict[str, Any] | None":
    """Fetch one pod by id (None if not found)."""

    try:
        pod = _api_request("GET", f"/pods/{pod_id}", api_key, timeout=30)
    except RunPodError:
        return None
    return pod if isinstance(pod, dict) else None


def terminate_pod(api_key: str, pod_id: str) -> "dict[str, Any]":
    """Delete (terminate) a pod by id — the cost-saving action for an idle pod.

    Uses the same ``DELETE /pods/{id}`` endpoint as the launcher's delete watchdog.
    A 404 (already gone) is treated as success.
    """

    try:
        _api_request("DELETE", f"/pods/{pod_id}", api_key, timeout=60)
        return {"id": pod_id, "terminated": True, "detail": "deleted"}
    except RunPodError as exc:
        if "HTTP 404" in str(exc):
            return {"id": pod_id, "terminated": True, "detail": "already gone (404)"}
        return {"id": pod_id, "terminated": False, "detail": str(exc)}


def restart_pod(api_key: str, pod_id: str) -> "dict[str, Any]":
    """Recover a stalled pod: stop then start it. Returns a small result dict."""

    steps: "list[str]" = []
    for verb in ("stop", "start"):
        try:
            _api_request("POST", f"/pods/{pod_id}/{verb}", api_key, timeout=60)
            steps.append(f"{verb}=ok")
        except RunPodError as exc:
            steps.append(f"{verb}=error:{exc}")
            return {"id": pod_id, "restarted": False, "steps": steps}
    return {"id": pod_id, "restarted": True, "steps": steps}


def check(api_key: str, *, restart_stalled: bool = False) -> "dict[str, Any]":
    """List pods, classify each, optionally restart stalled ones."""

    pods = _list_pods(api_key)
    verdicts = [classify_pod(p) for p in pods]
    stalled = [v for v in verdicts if v["verdict"] == "stalled"]
    actions: "list[dict[str, Any]]" = []
    if restart_stalled and stalled:
        actions = [restart_pod(api_key, v["id"]) for v in stalled]
    return {
        "connected": True,
        "pod_count": len(verdicts),
        "counts": {
            "running": sum(v["verdict"] == "running" for v in verdicts),
            "stalled": len(stalled),
            "stopped": sum(v["verdict"] == "stopped" for v in verdicts),
            "unknown": sum(v["verdict"] == "unknown" for v in verdicts),
        },
        "pods": verdicts,
        "stalled": stalled,
        "restart_actions": actions,
    }


def _print_human(result: "dict[str, Any]") -> None:
    c = result["counts"]
    print(f"[runpod] connected — {result['pod_count']} pod(s): "
          f"{c['running']} running, {c['stalled']} stalled, "
          f"{c['stopped']} stopped, {c['unknown']} unknown")
    for v in result["pods"]:
        mark = "⚠ STALLED" if v["verdict"] == "stalled" else v["verdict"]
        print(f"  - {v['name']} ({v['id']}): {mark} — {v['reason']}")
    for a in result.get("restart_actions", []):
        state = "restarted" if a["restarted"] else "RESTART FAILED"
        print(f"  → {a['id']}: {state} [{', '.join(a['steps'])}]")


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="List pods and flag stalled ones (needs a key)")
    parser.add_argument("--restart-stalled", action="store_true",
                        help="Stop+start any stalled pod found by --check")
    parser.add_argument("--pod", default=None, metavar="POD_ID",
                        help="Inspect a single pod by id (status/shape verdict)")
    parser.add_argument("--terminate", default=None, metavar="POD_ID",
                        help="Terminate (delete) an idle/unused pod by id to save cost. "
                             "Requires --yes. REST cannot prove idleness (no GPU-util "
                             "telemetry); the pod's status is printed before deletion.")
    parser.add_argument("--yes", action="store_true",
                        help="Confirm a destructive action (--terminate)")
    parser.add_argument("--api-key", default=None,
                        help="Explicit key (prefer RUNPOD_API_KEY env; never commit one)")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument("--dry-run", action="store_true",
                        help="No network: report which connection route is available")
    args = parser.parse_args(argv)

    key, source = resolve_api_key(args.api_key)

    if args.dry_run:
        out = {
            "route": "direct" if key else "github-mediated",
            "key_present": bool(key),
            "key_source": source,
            "github_workflow": DISPATCH_WORKFLOW,
            "hint": ("RUNPOD_API_KEY present — direct REST connection available."
                     if key else GITHUB_FALLBACK_HINT),
        }
        print(json.dumps(out, indent=2) if args.json else out["hint"])
        return 0

    if not key:
        # Fail closed with an actionable, always-available fallback.
        if args.json:
            print(json.dumps({"connected": False, "key_present": False,
                              "route": "github-mediated",
                              "github_workflow": DISPATCH_WORKFLOW,
                              "hint": GITHUB_FALLBACK_HINT}, indent=2))
        else:
            print(GITHUB_FALLBACK_HINT, file=sys.stderr)
        return 2

    if args.pod:
        pod = get_pod(key, args.pod)
        if pod is None:
            out = {"connected": True, "found": False, "id": args.pod}
            print(json.dumps(out, indent=2) if args.json else
                  f"[runpod] pod {args.pod} not found")
            return 1
        verdict = classify_pod(pod)
        print(json.dumps(verdict, indent=2) if args.json else
              f"[runpod] {verdict['name']} ({verdict['id']}): "
              f"{verdict['verdict']} — {verdict['reason']}")
        return 0

    if args.terminate:
        pod = get_pod(key, args.terminate)
        verdict = classify_pod(pod) if pod else {"id": args.terminate, "verdict": "not-found",
                                                 "reason": "pod not found"}
        if not args.yes:
            out = {"terminated": False, "blocked": "needs --yes",
                   "pod": verdict,
                   "note": ("REST cannot prove a pod is idle (no GPU-util telemetry). "
                            "Re-run with --yes to delete; this stops billing immediately.")}
            print(json.dumps(out, indent=2) if args.json else
                  f"[runpod] WOULD terminate {verdict['id']} "
                  f"(status: {verdict.get('reason')}). Re-run with --yes to confirm.")
            return 2
        result = terminate_pod(key, args.terminate)
        result["pod_before"] = verdict
        print(json.dumps(result, indent=2) if args.json else
              f"[runpod] terminate {result['id']}: "
              f"{'OK' if result['terminated'] else 'FAILED'} — {result['detail']}")
        return 0 if result["terminated"] else 1

    if not args.check:
        msg = f"[runpod] key resolved from {source}; pass --check to query pods."
        print(json.dumps({"connected": True, "key_source": source}) if args.json else msg)
        return 0

    try:
        result = check(key, restart_stalled=args.restart_stalled)
    except RunPodError as exc:
        if args.json:
            print(json.dumps({"connected": False, "error": str(exc)}, indent=2))
        else:
            print(f"[runpod] connection failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)
    # Non-zero exit when something is stalled and we were not asked to fix it,
    # so CI / callers notice. Zero if we restarted them.
    if result["counts"]["stalled"] and not args.restart_stalled:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

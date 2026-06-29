#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Manual-only RunPod launcher for the Prosoche Focus-Efficiency-Frontier (thesis §5).

The "RunPod through GitHub" path, mirroring ``tools/runpod_rlvr.py`` (whose SSH/REST/
watchdog/pod-delete plumbing this reuses). It is dispatch-only and money-spending, so
it never runs on push/PR/schedule. On dispatch it rents a GPU pod via the RunPod REST
API, syncs the repo, runs the focus-frontier harness over SSH, copies the report back,
and ALWAYS deletes the pod (finally block + the remote watchdog inherited from
``runpod_rlvr``).

Honesty contract (this is the important part):
  * ``--local``       : run the harness on THIS machine (no pod, no cost) — the cheap
                        smoke that validates the whole path before any spend.
  * ``--dry-run``     : print the sanitized pod payload + remote command (no pod).
  * ``--yes``         : rent a pod and run the harness on it.
  * ``--remote-mode`` : ``offline`` runs the deterministic mechanism + routing harness
                        (NO-GO by design — what ships today). ``live`` ALSO runs it, and
                        loudly states that the *measured* 3-arm token-per-solved-task eval
                        (live model + >=2 judge families over a decontaminated task set) is
                        the model-gated OPEN item — it does NOT fabricate a number. When
                        that eval loop lands, point ``--eval-entrypoint`` at it here.

A pod always costs real money while it runs. The RunPod key is read from
``RUNPOD_API_KEY``, never printed (redacted via ``runpod_rlvr._redact``), and the pod is
deleted in a ``finally`` block plus a remote watchdog. canClaimAGI:false.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The repo-sanctioned reuse (see runpod_rlvr.py line ~194): share one set of pod
# plumbing across launchers rather than re-implementing the REST/SSH/watchdog logic.
from tools.runpod_rlvr import (  # noqa: E402
    RunPodError,
    _api_request,
    _build_create_payload,
    _delete_pod,
    _generate_ssh_key,
    _pod_id,
    _poll_ssh,
    _rsync_repo_to_pod,
    _scp_from_pod,
    _ssh_base,
    _stream,
    _wait_ssh_login,
)

REPORT_REL = "agi-proof/benchmark-results/prosoche/focus-efficiency.PENDING.public-report.json"
ROBUST_REL = "agi-proof/benchmark-results/prosoche/prosoche-robustness.json"
EVAL_REL = "agi-proof/benchmark-results/prosoche/focus-frontier-eval.PENDING.public-report.json"
POWERED_REL = "agi-proof/benchmark-results/prosoche/focus-frontier-eval.powered-public.json"
REMOTE_REPO = "/workspace/sophia-runpod/sophia-agi"


def _remote_focus_script(args: argparse.Namespace) -> str:
    """The bash run over SSH on the pod. Deterministic + honest: it runs the harness
    and copies the report back; it never invents a measured effect."""
    # The 3-arm eval entrypoint: deterministic survival-proxy by default (NO-GO,
    # exercises the math). On the farm, pass --model <spec> to it for the real run.
    # SECURITY: ``eval_entrypoint`` and ``model`` are interpolated into the bash script
    # run over SSH, so shell-escape them with shlex.quote (mirrors tools/runpod_rlvr.py)
    # to close the shell-injection sink — never interpolate raw dispatch input.
    entry = shlex.quote(args.eval_entrypoint or "tools/run_focus_frontier_eval.py")
    live_cmd = f"python {entry} --write"
    if args.remote_mode == "live":
        live_cmd = (
            'echo "[focus-frontier] LIVE: the MEASURED 3-arm eval (real model + >=2 judge '
            'families) is the model-gated OPEN item (agi-proof/failure-ledger.md: '
            'prosoche-efficiency-token-saving). This validates the pod path + the eval '
            'machinery and returns the honest NO-GO report — no number is fabricated."\n'
            f"python {entry} --write"
        )
        if args.model:
            live_cmd += f' --model {shlex.quote(args.model)} || echo "[focus-frontier] real-model arm refused/failed; honest NO-GO report stands"'
    # On the farm, score the POWERED public split with a real subject + >= 2 judge
    # families (env keys/models supplied to the pod) when in live mode.
    powered = ""
    if args.remote_mode == "live" and args.model:
        powered = (
            f'python {entry} --real --split public --max-workers 8 '
            f'--out agi-proof/benchmark-results/prosoche/focus-frontier-eval.powered-public.json '
            f'|| echo "[focus-frontier] powered run failed; honest reports stand"'
        )
    return f"""
set -Eeuo pipefail
cd {REMOTE_REPO}
echo "[focus-frontier] python: $(python --version 2>&1)"
python tools/build_focus_battery.py --check
python tools/run_focus_efficiency_frontier.py --write
python tools/run_focus_efficiency_frontier.py --check
python tools/run_prosoche_robustness.py --write
{live_cmd}
python {entry} --check
{powered}
echo "[focus-frontier] done; reports written."
"""


def _run_local(args: argparse.Namespace) -> int:
    """No pod, no cost — run the harness on this machine to validate the path."""
    from tools.run_focus_efficiency_frontier import build_report, REPORT
    import json

    report = build_report()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[local] wrote {REPORT.relative_to(ROOT)}  verdict={report['verdict']} "
          f"routingAccuracy={report['routingFidelity']['routingAccuracy']}")
    if args.remote_mode == "live":
        print("[local] NOTE: live measured eval is the model-gated OPEN item; local smoke is "
              "the NO-GO mechanism report only (no fabricated number).")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="run the harness locally (no pod, no cost)")
    ap.add_argument("--dry-run", action="store_true", help="print sanitized payload + remote command; create no pod")
    ap.add_argument("--yes", action="store_true", help="actually create a RunPod pod (required unless --dry-run/--local)")
    ap.add_argument("--remote-mode", choices=["offline", "live"], default="offline",
                    help="offline = deterministic mechanism/routing harness (NO-GO by design); live = same + state the model-gated measured eval is open")
    ap.add_argument("--eval-entrypoint", default="tools/run_focus_frontier_eval.py",
                    help="the 3-arm eval tool run on the pod (--write emits the gated artifact)")
    ap.add_argument("--model", default="",
                    help="(live) real subject model spec passed to the eval entrypoint for the measured run")
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    ap.add_argument("--name", default=f"sophia-focus-frontier-{ts}")
    ap.add_argument("--branch", default="", help="git branch/tag to run (the pod clones the repo)")
    ap.add_argument("--gpu-type", default="NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA H100 80GB HBM3,NVIDIA H100 PCIe")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud-type", default="SECURE")
    ap.add_argument("--image-name", default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    ap.add_argument("--container-disk-gb", type=int, default=40)
    ap.add_argument("--volume-gb", type=int, default=40)
    ap.add_argument("--interruptible", action="store_true")
    ap.add_argument("--network-volume-id", default="")
    ap.add_argument("--allowed-cuda-versions", default="")
    ap.add_argument("--no-remote-delete-watchdog", action="store_true",
                    help="disable the on-pod self-delete watchdog (NOT recommended)")
    ap.add_argument("--auto-exit-seconds", type=int, default=3600)
    ap.add_argument("--ssh-timeout-s", type=int, default=600)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)

    if args.local:
        return _run_local(args)

    api_key = os.environ.get(args.api_key_env, "")
    if args.dry_run:
        # Build the payload with a placeholder key/pubkey — no network, no pod.
        payload = _build_create_payload(args, public_key="ssh-ed25519 AAAA...DRYRUN", api_key="")
        from tools.runpod_rlvr import _redact
        print("[dry-run] RunPod create payload (sanitized):")
        # never print env secrets
        payload.get("env", {}).pop("RUNPOD_API_KEY", None)
        import json
        print(json.dumps(payload, indent=2)[:2000])
        print("\n[dry-run] remote command:")
        print(_remote_focus_script(args))
        print(f"[dry-run] api key present: {bool(api_key)} ({_redact(api_key)})")
        return 0

    if not args.yes:
        print("::error:: refusing to create a paid pod without --yes (use --local or --dry-run for no-cost paths)",
              file=sys.stderr)
        return 2
    if not api_key:
        print(f"::error:: {args.api_key_env} is empty; add a valid RunPod key. Aborting (no pod created).",
              file=sys.stderr)
        return 2

    import tempfile

    pod_id = ""
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        key_path, public_key = _generate_ssh_key(tmpdir)
        payload = _build_create_payload(args, public_key, api_key=api_key)
        try:
            pod = _api_request("POST", "/pods", api_key, payload)
            pod_id = _pod_id(pod)
            print(f"[focus-frontier] pod {pod_id} created; connecting…")
            conn = _poll_ssh(api_key, pod_id, timeout_s=args.ssh_timeout_s)
            _wait_ssh_login(conn, key_path)
            _rsync_repo_to_pod(conn, key_path)
            log_path = tmpdir / "focus-frontier.log"
            cmd = _ssh_base(conn, key_path) + ["bash", "-s"]
            exit_code = _stream(cmd, log_path, input_text=_remote_focus_script(args))
            for rel in (REPORT_REL, ROBUST_REL, EVAL_REL, POWERED_REL):
                _scp_from_pod(conn, key_path, f"{REMOTE_REPO}/{rel}", ROOT / rel)
            print(f"[focus-frontier] remote exit={exit_code}; reports copied back.")
            return exit_code
        except RunPodError as exc:
            print(f"::error:: RunPod error: {exc}", file=sys.stderr)
            return 1
        finally:
            if pod_id:
                try:
                    _delete_pod(api_key, pod_id)
                    print(f"[focus-frontier] pod {pod_id} deleted.")
                except Exception as exc:  # noqa: BLE001
                    print(f"::warning:: failed to delete pod {pod_id}: {exc}; reap with tools/runpod_connect.py --reap-exited",
                          file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

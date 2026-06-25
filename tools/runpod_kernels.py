#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Build + roofline-profile HPC kernels on a rented RunPod CUDA GPU.

This is the M1 orchestrator for the HPC operator track
(docs/06-Roadmap/HPC-Operator-Compiler-Roadmap.md). It reuses the *proven* pod lifecycle
from tools/runpod_rlvr.py (create pod -> poll SSH -> run over SSH -> copy artifacts back
-> ALWAYS delete pod) and runs the roofline harness against real kernels, profiling SM and
memory utilization with Nsight Compute (``ncu``) when available.

Default is --dry-run: NO pod, NO cost. It prints the exact remote script that *would* run,
so the plumbing is reviewable offline and in CI before a single GPU-second is spent.

    python tools/runpod_kernels.py --dry-run
    RUNPOD_API_KEY=... python tools/runpod_kernels.py --yes --branch <branch> --gpu-type "NVIDIA H100 80GB HBM3"

Honest status: as of M1 the only kernel that exists is the *roofline harness self-test*
plus the offline synthetic demo. This orchestrator is wired and dry-run-correct so that the
first real Triton/CUDA GEMM can be dropped into ``kernels/src`` and profiled the moment it
lands — born already measured against the physical limit.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the proven RunPod lifecycle. Nothing here re-implements pod create/SSH/teardown.
from tools.runpod_rlvr import (  # noqa: E402
    DEFAULT_REPO_URL,
    RunPodError,
    _api_request,
    _build_create_payload,
    _delete_pod,
    _generate_ssh_key,
    _poll_ssh,
    _pod_id,
    _rsync_repo_to_pod,
    _scp_from_pod,
    _ssh_base,
    _startup_cmd,
    _stream,
    _wait_ssh_login,
)


def _remote_kernel_script(args: argparse.Namespace) -> str:
    """The script run on the pod: clone/refresh repo, run roofline, optionally ncu-profile.

    Kept deliberately small and idempotent. The roofline harness needs no GPU for its
    self-test, so this stays green even before any CUDA kernel exists; the GPU is used for
    the (optional) ncu profiling step and, later, for real kernel timing.
    """
    branch = args.branch
    return f"""
set -Eeuo pipefail
cd /workspace
if [ ! -d sophia-agi ]; then
  git clone --depth 1 --branch {branch} {DEFAULT_REPO_URL} sophia-agi
fi
cd sophia-agi
git fetch --depth 1 origin {branch} || true
git checkout {branch} || true
git pull --ff-only origin {branch} || true

mkdir -p kernels/reports
echo "== nvidia-smi =="
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv || true

echo "== roofline self-test (no GPU needed; proves the gate runs) =="
python kernels/bench/roofline.py --self-test | tee kernels/reports/roofline_selftest.txt

DEV="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/^ *//;s/ *$//')"
echo "== roofline demo for detected device: $DEV =="
python kernels/bench/roofline.py --demo --device "$DEV" --json \\
  | tee kernels/reports/roofline_demo.json || true

# --- M1+ hook: when a real kernel exists at kernels/src, build + ncu-profile it here. ---
if [ -f kernels/src/run_kernel.py ]; then
  echo "== ncu profile of kernels/src/run_kernel.py =="
  if command -v ncu >/dev/null 2>&1; then
    ncu --set full --target-processes all \\
      -o kernels/reports/ncu_profile \\
      python kernels/src/run_kernel.py || true
  else
    echo "ncu not found in image; running kernel un-profiled"
    python kernels/src/run_kernel.py | tee kernels/reports/kernel_run.txt || true
  fi
else
  echo "no kernels/src/run_kernel.py yet — roofline gate verified, kernel TODO (M1)."
fi

echo "== artifacts =="
ls -la kernels/reports
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="print the remote script and exit; no pod, no cost (DEFAULT)")
    p.add_argument("--yes", dest="dry_run", action="store_false",
                   help="actually rent a pod and run (requires RUNPOD_API_KEY)")
    p.add_argument("--branch", default="claude/hpc-operator-compiler-roadmap-zumu83",
                   help="repo branch to clone/run on the pod")
    p.add_argument("--name", default="sophia-kernels", help="pod name")
    p.add_argument("--gpu-type", default="NVIDIA H100 80GB HBM3,NVIDIA A100-SXM4-80GB",
                   help="comma-separated RunPod GPU type priority list")
    p.add_argument("--cloud-type", default="SECURE")
    p.add_argument("--ssh-timeout", type=int, default=900, help="seconds to wait for SSH")
    p.add_argument("--auto-exit-seconds", type=int, default=3600,
                   help="pod self-destruct guard if the orchestrator dies")
    p.add_argument("--no-remote-delete-watchdog", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    remote_script = _remote_kernel_script(args)

    if args.dry_run:
        print("=== DRY RUN — no pod will be created, no cost incurred ===")
        print(f"GPU priority : {args.gpu_type}")
        print(f"branch       : {args.branch}")
        print("\n--- remote script that WOULD run over SSH on the pod ---")
        print(remote_script)
        print("--- end remote script ---")
        print("\nRun for real with:  RUNPOD_API_KEY=... python tools/runpod_kernels.py --yes")
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY not set; refusing to create a pod.", file=sys.stderr)
        return 2

    pod_id: str | None = None
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        key_path, public_key = _generate_ssh_key(tmp)
        try:
            payload = _build_create_payload(args, public_key, api_key)
            payload["dockerStartCmd"] = _startup_cmd(args.auto_exit_seconds)
            created = _api_request("POST", "/pods", api_key, payload)
            pod_id = _pod_id(created)
            print(f"created pod {pod_id}; waiting for SSH…")
            conn = _poll_ssh(api_key, pod_id, timeout_s=args.ssh_timeout)
            _wait_ssh_login(conn, key_path)
            _rsync_repo_to_pod(conn, key_path)

            log_path = tmp / "kernels_run.log"
            ssh = _ssh_base(conn, key_path)
            rc = _stream(ssh + ["bash", "-lc", remote_script], log_path)

            # Copy reports back regardless of rc so a partial run is still inspectable.
            out_dir = ROOT / "kernels" / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in ("roofline_selftest.txt", "roofline_demo.json"):
                _scp_from_pod(conn, key_path, f"/workspace/sophia-agi/kernels/reports/{name}",
                              out_dir / name)
            print(f"remote exit code: {rc}; reports in {out_dir}")
            return rc
        except RunPodError as exc:
            print(f"RunPod error: {exc}", file=sys.stderr)
            return 1
        finally:
            if pod_id:
                print(f"deleting pod {pod_id} (always, even on failure)…")
                _delete_pod(api_key, pod_id)


if __name__ == "__main__":
    sys.exit(main())

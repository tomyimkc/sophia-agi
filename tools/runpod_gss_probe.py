#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""RunPod launcher for the GSS probe — a real go/no-go on a real MoE checkpoint.

Rents a GPU pod via the RunPod REST API, runs ``tools/gss_probe.py --backend hf`` on a
real Mixture-of-Experts checkpoint over SSH (full-precision pass + a 4-bit self-draft →
ρ, α, k → the Governed-Speculative-Sparsity cost ratio), copies the JSON report back into
``agi-proof/benchmark-results/``, and **always deletes the pod** (finally block + the
remote watchdog). The orchestration mirrors ``tools/runpod_kernels.py`` exactly and reuses
the same proven helpers from ``tools/runpod_rlvr.py``; the secret is read from
``RUNPOD_API_KEY`` and never printed.

    # no pod, no cost — just print the remote script:
    python tools/runpod_gss_probe.py --dry-run --branch <branch>
    # rent a GPU and run for real (needs RUNPOD_API_KEY):
    RUNPOD_API_KEY=... python tools/runpod_gss_probe.py --yes --branch <branch> \
        --model allenai/OLMoE-1B-7B-0924 --draft bnb

Honest scope: the report is a *feasibility* verdict (ρ, k, cost_ratio → GO/NO-GO), never
a speedup. A first-party single run on one checkpoint; the no-overclaim discipline
(≥3 runs, CIs) applies before any headline. ``canClaimAGI`` stays ``false``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.runpod_rlvr import (  # noqa: E402
    DEFAULT_REPO_URL,
    RunPodError,
    _api_request,
    _build_create_payload,
    _delete_pod,
    _generate_ssh_key,
    _poll_ssh,
    _pod_id,
    _scp_from_pod,
    _ssh_base,
    _startup_cmd,
    _stream,
    _wait_ssh_login,
)

REMOTE_REPORT_DIR = "/workspace/sophia-agi/agi-proof/benchmark-results"


def _safe_slug(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "-" for c in model).strip("-")


def _remote_script(args: argparse.Namespace) -> str:
    """The script the pod runs over SSH: refresh repo, install deps, run the probe."""
    branch_q = shlex.quote(args.branch)
    model_q = shlex.quote(args.model)
    draft_q = shlex.quote(args.draft)
    slug = _safe_slug(args.model)
    report = f"{REMOTE_REPORT_DIR}/gss-{slug}.json"
    stdout = f"{REMOTE_REPORT_DIR}/gss-{slug}.stdout.txt"
    tokens, gamma = args.tokens, args.gamma
    coverage = args.coverage
    return f"""
set -Eeuo pipefail
BRANCH={branch_q}
cd /workspace
if [ ! -d sophia-agi ]; then
  git clone --depth 1 --branch "$BRANCH" {DEFAULT_REPO_URL} sophia-agi
fi
cd sophia-agi
git fetch --depth 1 origin "$BRANCH"
git checkout "$BRANCH"
git reset --hard "origin/$BRANCH"
mkdir -p {shlex.quote(REMOTE_REPORT_DIR)}

echo "== nvidia-smi =="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "== install probe deps (image ships torch+CUDA) =="
python -c "import numpy" 2>/dev/null || pip install -q numpy
python -c "import transformers" 2>/dev/null || pip install -q "transformers>=4.45" accelerate
python -c "import bitsandbytes" 2>/dev/null || pip install -q bitsandbytes

echo "== GSS probe: {model_q} (draft={draft_q}, tokens={tokens}) =="
python tools/gss_probe.py --backend hf \\
  --model {model_q} --draft {draft_q} \\
  --tokens {tokens} --gamma {gamma} --coverage {coverage} \\
  --out {shlex.quote(report)} 2>&1 | tee {shlex.quote(stdout)}

echo "== report =="
cat {shlex.quote(report)} || echo "no report written (see stdout above)"
ls -la {shlex.quote(REMOTE_REPORT_DIR)}
"""


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dry-run", action="store_true", default=True,
                   help="print the remote script and exit; no pod, no cost (DEFAULT)")
    p.add_argument("--yes", dest="dry_run", action="store_false",
                   help="actually rent a pod and run (requires RUNPOD_API_KEY)")
    p.add_argument("--branch", default="main", help="repo branch to clone/run on the pod")
    p.add_argument("--name", default="sophia-gss-probe", help="pod name")
    # probe knobs
    p.add_argument("--model", default="allenai/OLMoE-1B-7B-0924",
                   help="HF MoE checkpoint (router-logits-capable)")
    p.add_argument("--draft", choices=["bnb", "fakequant"], default="bnb",
                   help="4-bit self-draft source")
    p.add_argument("--tokens", type=int, default=128, help="max positions to score")
    p.add_argument("--gamma", type=int, default=4, help="speculative block size")
    p.add_argument("--coverage", type=float, default=0.9, help="read-set mass coverage")
    # pod shape (same namespace fields _build_create_payload expects)
    p.add_argument("--gpu-type", default="NVIDIA A40,NVIDIA L40S,NVIDIA RTX A6000,NVIDIA GeForce RTX 4090",
                   help="comma-separated RunPod GPU type priority list")
    p.add_argument("--gpu-count", type=int, default=1)
    p.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    p.add_argument("--interruptible", action="store_true", help="cheaper spot pod")
    p.add_argument("--image-name", default="runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2204",
                   help="container image; ships torch+CUDA")
    p.add_argument("--container-disk-gb", type=int, default=60)
    p.add_argument("--volume-gb", type=int, default=40)
    p.add_argument("--allowed-cuda-versions", default="")
    p.add_argument("--ssh-timeout", type=int, default=900, help="seconds to wait for SSH")
    p.add_argument("--auto-exit-seconds", type=int, default=2400,
                   help="pod self-destruct guard if the orchestrator dies")
    p.add_argument("--no-remote-delete-watchdog", action="store_true")
    return p.parse_args(argv)


def main(argv: "list[str] | None" = None) -> int:
    args = parse_args(argv)
    remote_script = _remote_script(args)

    if args.dry_run:
        print("=== DRY RUN — no pod will be created, no cost incurred ===")
        print(f"model        : {args.model}  (draft={args.draft})")
        print(f"GPU priority : {args.gpu_type}")
        print(f"branch       : {args.branch}")
        print("\n--- remote script that WOULD run over SSH on the pod ---")
        print(remote_script)
        print("--- end remote script ---")
        print("\nRun for real with:  RUNPOD_API_KEY=... python tools/runpod_gss_probe.py --yes")
        return 0

    api_key = os.environ.get("RUNPOD_API_KEY", "")
    if not api_key:
        print("RUNPOD_API_KEY not set; refusing to create a pod.", file=sys.stderr)
        return 2

    slug = _safe_slug(args.model)
    pod_id: "str | None" = None
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

            log_path = tmp / "gss_run.log"
            ssh = _ssh_base(conn, key_path)
            rc = _stream(ssh + ["bash", "-lc", remote_script], log_path)

            out_dir = ROOT / "agi-proof" / "benchmark-results"
            out_dir.mkdir(parents=True, exist_ok=True)
            for name in (f"gss-{slug}.json", f"gss-{slug}.stdout.txt"):
                _scp_from_pod(conn, key_path, f"{REMOTE_REPORT_DIR}/{name}", out_dir / name)
            print(f"remote exit code: {rc}; report in {out_dir}/gss-{slug}.json")
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

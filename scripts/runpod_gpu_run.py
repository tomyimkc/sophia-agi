#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Provision a RunPod GPU pod, run the sophia-kvcache GPU HBM smoke test on real
device memory, stream its result, and tear the pod down.

The smoke binary is built on a GPU-less CI runner (the `cuda` feature uses
cudarc `dynamic-loading`, which dlopens libcuda at runtime), uploaded to the pod
over its base64-embedded startup command, and executed against the pod's GPU.

Env:
  RUNPOD_API_KEY  (required)  RunPod API key
  GPU_TYPE        GPU type id (default: NVIDIA GeForce RTX 4090)
  BLOCKS          blocks to round-trip   (default 2048)
  BLOCK_BYTES     bytes per block        (default 262144)
  CUDA_IMAGE      pod image (default nvidia/cuda:12.4.1-runtime-ubuntu22.04)
  MAX_WAIT_SECS   provisioning timeout   (default 600)

Usage: runpod_gpu_run.py <path-to-gpu_hbm_smoke-binary>
"""
import base64
import os
import sys
import time

try:
    import runpod
except ImportError:
    sys.exit("runpod SDK not installed: pip install 'runpod>=1.6'")

SENTINEL_OK = "RESULT: PASS"


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: runpod_gpu_run.py <gpu_hbm_smoke binary>")
    binary_path = sys.argv[1]

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        sys.exit("RUNPOD_API_KEY not set")
    runpod.api_key = api_key

    gpu_type = os.environ.get("GPU_TYPE", "NVIDIA GeForce RTX 4090")
    image = os.environ.get("CUDA_IMAGE", "nvidia/cuda:12.4.1-runtime-ubuntu22.04")
    blocks = os.environ.get("BLOCKS", "2048")
    block_bytes = os.environ.get("BLOCK_BYTES", "262144")
    max_wait = int(os.environ.get("MAX_WAIT_SECS", "600"))

    with open(binary_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode()

    # The pod decodes the binary, runs it against the GPU, and prints the result
    # (captured in the pod's container logs / RunPod console).
    start_cmd = (
        "bash -lc '"
        f"echo {b64} | base64 -d > /tmp/gpu_hbm_smoke && chmod +x /tmp/gpu_hbm_smoke && "
        f"BLOCKS={blocks} BLOCK_BYTES={block_bytes} /tmp/gpu_hbm_smoke; "
        "echo SMOKE_EXIT=$?; sleep 5'"
    )

    print(f"[runpod] creating pod: gpu='{gpu_type}' image='{image}'")
    pod = runpod.create_pod(
        name="sophia-kvcache-gpu-smoke",
        image_name=image,
        gpu_type_id=gpu_type,
        gpu_count=1,
        container_disk_in_gb=10,
        docker_args=start_cmd,
        cloud_type="SECURE",
    )
    pod_id = pod["id"]
    print(f"[runpod] pod {pod_id} created; waiting for it to run/complete")

    ok = False
    try:
        deadline = time.time() + max_wait
        while time.time() < deadline:
            info = runpod.get_pod(pod_id)
            status = (info or {}).get("desiredStatus", "?")
            runtime = (info or {}).get("runtime")
            print(f"[runpod] status={status} runtime={'up' if runtime else 'pending'}")
            # Logs are surfaced in the RunPod console; the CI step also prints
            # the binary's stdout via the pod's log stream when available.
            logs = _logs(pod_id)
            if logs:
                print(logs)
                if SENTINEL_OK in logs:
                    ok = True
                    break
                if "SMOKE_EXIT=" in logs and SENTINEL_OK not in logs:
                    break
            time.sleep(15)
    finally:
        print(f"[runpod] terminating pod {pod_id}")
        runpod.terminate_pod(pod_id)

    if ok:
        print("[runpod] GPU HBM smoke PASSED on real device memory")
        return 0
    print("[runpod] GPU HBM smoke did not report PASS (see logs above)")
    return 1


def _logs(pod_id: str) -> str:
    """Best-effort pod log fetch. The RunPod SDK's log surface varies by version;
    fall back to empty (the RunPod console always has the full stream)."""
    try:
        return runpod.get_pod_logs(pod_id)  # type: ignore[attr-defined]
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())

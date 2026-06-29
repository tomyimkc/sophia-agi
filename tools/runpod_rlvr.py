#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Launch Sophia's live RLVR run on RunPod, then always terminate the Pod.

This is intentionally a small, dependency-free orchestrator:

1. Create a one-shot RunPod GPU Pod through the REST API.
2. Start SSH inside the Pod using an ephemeral key generated locally.
3. Stream the RLVR training command over SSH.
4. Copy back the public report if it exists.
5. DELETE the Pod in a ``finally`` block unless ``--keep-pod`` is explicitly set.
6. Also install a remote delete watchdog by default, so if the local Mac/SSH
   session dies the Pod is deleted after ``--auto-exit-seconds``.

Security notes:

- Do not hard-code API keys in this file or in shell history. Set
  ``RUNPOD_API_KEY`` in your environment.
- The script redacts the API key from its own logs.
- It does not write the API key to the repository.

Example:

    export RUNPOD_API_KEY='rpa_...'
    python3 tools/runpod_rlvr.py --yes

Dry-run the RunPod payload without renting a GPU:

    python3 tools/runpod_rlvr.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNPOD_API_BASE = "https://rest.runpod.io/v1"
# Modern RunPod PyTorch image with torch already installed. This avoids a huge
# pip-side torch/CUDA download during live RLVR bootstrap.
DEFAULT_IMAGE = "runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2204"
DEFAULT_REPO_URL = "https://github.com/tomyimkc/sophia-agi.git"
DEFAULT_GPU_TYPES = [
    # 1x80GB bf16 path is the least finicky for this repo's live GRPO runner.
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H100 PCIe",
]


class RunPodError(RuntimeError):
    """Raised for RunPod API or orchestration failures."""


@dataclass
class PodConnection:
    pod_id: str
    public_ip: str
    ssh_port: int


def _redact(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 12:
        return "***"
    return f"{value[:6]}…{value[-4:]}"


def _api_request(
    method: str,
    path: str,
    api_key: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: int = 60,
) -> Any:
    data = None
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"{RUNPOD_API_BASE}{path}",
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RunPodError(f"RunPod API {method} {path} failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RunPodError(f"RunPod API {method} {path} failed: {exc}") from exc


def _startup_cmd(auto_exit_seconds: int) -> list[str]:
    """Return a pod start command that only starts SSH and idles.

    The local orchestrator runs the real training command over SSH so it can
    stream logs, copy artifacts, and delete the Pod in ``finally``. ``timeout``
    is a safety guard: if the local process disappears, the container exits
    after the configured window rather than idling forever.
    """

    script = f"""
set -Eeuo pipefail
cleanup() {{
  code=$?
  # Derive our own pod id robustly: prefer RunPod's injected RUNPOD_POD_ID, but
  # fall back to the container hostname (RunPod sets it to the pod id). Relying on
  # RUNPOD_POD_ID alone silently leaked EXITED pods when it was absent.
  POD_ID="${{RUNPOD_POD_ID:-$(hostname)}}"
  if [ "${{SOPHIA_REMOTE_DELETE_WATCHDOG:-0}}" = "1" ] && [ -n "${{RUNPOD_API_KEY:-}}" ] && [ -n "$POD_ID" ]; then
    echo "Sophia watchdog deleting RunPod Pod $POD_ID after startup command exit (code $code)."
    if ! curl -fsS --request DELETE \\
      --url "https://rest.runpod.io/v1/pods/${{POD_ID}}" \\
      --header "Authorization: Bearer $RUNPOD_API_KEY"; then
      # Non-fatal, but never silent: a swallowed failure here is exactly how pods leak.
      echo "[sophia-watchdog] WARNING: failed to delete pod $POD_ID; it may linger and bill — reap it with tools/runpod_connect.py --reap-exited" >&2
    fi
  else
    echo "[sophia-watchdog] WARNING: watchdog not armed (watchdog=${{SOPHIA_REMOTE_DELETE_WATCHDOG:-0}}, key=$([ -n "${{RUNPOD_API_KEY:-}}" ] && echo set || echo unset), pod_id=$([ -n "$POD_ID" ] && echo "$POD_ID" || echo empty)); pod will NOT self-delete." >&2
  fi
  exit "$code"
}}
trap cleanup EXIT
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends openssh-server git curl ca-certificates rsync
mkdir -p /var/run/sshd /root/.ssh
chmod 700 /root/.ssh
printf '%s\\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
sed -i 's/^#\\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
sed -i 's/^#\\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
service ssh start
echo "Sophia RunPod SSH ready; auto-exit in {auto_exit_seconds}s if not terminated earlier."
timeout {auto_exit_seconds}s sleep infinity
"""
    return ["bash", "-lc", script]


def _build_create_payload(args: argparse.Namespace, public_key: str, api_key: str = "") -> dict[str, Any]:
    gpu_types = [g.strip() for g in args.gpu_type.split(",") if g.strip()]
    env = {
        "PUBLIC_KEY": public_key,
        "SOPHIA_REMOTE_DELETE_WATCHDOG": "0" if args.no_remote_delete_watchdog else "1",
    }
    if not args.no_remote_delete_watchdog and api_key:
        env["RUNPOD_API_KEY"] = api_key
    # Faithfulness task: forward the entailment LLM key (+ optional LLMHub base url / CA)
    # into the pod env so the on-pod verify seam works. Read from the launcher's environment
    # (on the GH runner: repo Actions secrets), NEVER hardcoded. Only injected when present.
    for _var in ("DEEPSEEK_API_KEY", "LLMHUB_API_KEY", "LLMHUB_BASE_URL"):
        _val = os.environ.get(_var)
        if _val:
            env[_var] = _val
    payload: dict[str, Any] = {
        "name": args.name,
        "cloudType": args.cloud_type,
        "computeType": "GPU",
        "gpuTypeIds": gpu_types,
        "gpuTypePriority": "availability",
        "gpuCount": args.gpu_count,
        "imageName": args.image_name,
        "containerDiskInGb": args.container_disk_gb,
        "volumeInGb": args.volume_gb,
        "volumeMountPath": "/workspace",
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "interruptible": args.interruptible,
        "locked": False,
        "env": env,
        "dockerEntrypoint": [],
        "dockerStartCmd": _startup_cmd(args.auto_exit_seconds),
    }
    # getattr: _build_create_payload is reused by other launchers (e.g. runpod_nccl_bench.py) whose
    # arg parsers don't define --network-volume-id, so read it defensively.
    netvol = getattr(args, "network_volume_id", "")
    if netvol:
        # Persistent network volume mounted at /workspace (where HF_HOME lives): the model weight
        # cache then SURVIVES pod deletion, so subsequent runs skip the cold ~8GB+ download (billed
        # GPU minutes) — the saving the ephemeral volumeInGb above CANNOT give (it dies with the pod).
        # NB: a network volume is data-center-scoped, so the pod is pinned to that volume's DC (can
        # reduce GPU availability there). Only worth it past the break-even runs/mo — see
        # tools/runpod_volume_breakeven.py.
        payload["networkVolumeId"] = netvol
    if args.allowed_cuda_versions:
        payload["allowedCudaVersions"] = [
            v.strip() for v in args.allowed_cuda_versions.split(",") if v.strip()
        ]
    return payload


def _pod_id(pod: dict[str, Any]) -> str:
    pod_id = pod.get("id") or pod.get("podId")
    if not pod_id:
        raise RunPodError(f"RunPod create response did not include a pod id: {pod}")
    return str(pod_id)


def _poll_ssh(api_key: str, pod_id: str, timeout_s: int, interval_s: int = 10) -> PodConnection:
    deadline = time.time() + timeout_s
    last_status = ""
    while time.time() < deadline:
        try:
            pod = _api_request("GET", f"/pods/{pod_id}", api_key, timeout=30)
        except RunPodError as exc:
            print(f"[runpod] transient poll error for pod {pod_id}: {exc}; retrying")
            time.sleep(interval_s)
            continue
        public_ip = pod.get("publicIp")
        port_mappings = pod.get("portMappings") or {}
        ssh_port = (
            port_mappings.get("22")
            or port_mappings.get(22)
            or port_mappings.get("22/tcp")
            or port_mappings.get("tcp/22")
        )
        status = pod.get("desiredStatus") or pod.get("lastStatusChange") or "unknown"
        if public_ip and ssh_port:
            return PodConnection(pod_id=pod_id, public_ip=str(public_ip), ssh_port=int(ssh_port))
        if status != last_status:
            print(f"[runpod] waiting for SSH mapping; status={status}, ip={public_ip}, ports={port_mappings}")
            last_status = str(status)
        time.sleep(interval_s)
    raise RunPodError(f"Timed out waiting for public SSH mapping for pod {pod_id}")


def _run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[local] {printable}")
    proc = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.stdout:
        print(proc.stdout, end="")
    if check and proc.returncode != 0:
        raise RunPodError(f"Command failed with exit code {proc.returncode}: {printable}")
    return proc


def _stream(cmd: list[str], log_path: Path, *, input_text: str | None = None) -> int:
    printable = " ".join(shlex.quote(x) for x in cmd)
    print(f"[local] {printable}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdin=subprocess.PIPE if input_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if input_text is not None:
            assert proc.stdin is not None
            proc.stdin.write(input_text)
            proc.stdin.close()
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return proc.wait()


def _generate_ssh_key(tmpdir: Path) -> tuple[Path, str]:
    key_path = tmpdir / "runpod_sophia_ed25519"
    _run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-q",
            "-N",
            "",
            "-C",
            "sophia-runpod-ephemeral",
            "-f",
            str(key_path),
        ]
    )
    public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").strip()
    return key_path, public_key


def _ssh_base(conn: PodConnection, key_path: Path) -> list[str]:
    return [
        "ssh",
        "-i",
        str(key_path),
        "-p",
        str(conn.ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=20",
        f"root@{conn.public_ip}",
    ]


def _wait_ssh_login(conn: PodConnection, key_path: Path, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = _run(_ssh_base(conn, key_path) + ["echo sophia-ssh-ok"], check=False)
        if proc.returncode == 0 and "sophia-ssh-ok" in proc.stdout:
            return
        time.sleep(8)
    raise RunPodError(f"Timed out waiting for SSH login to {conn.public_ip}:{conn.ssh_port}")


def _remote_training_script(args: argparse.Namespace) -> str:
    # Avoid installing vLLM when explicitly not using it. This is the most robust
    # one-GPU path and avoids a large optional dependency.
    req_cmd = """
if [ "$SOPHIA_VLLM" = "none" ]; then
  grep -v '^vllm' requirements-rl.txt > /tmp/requirements-rl.sophia.txt
else
  cp requirements-rl.txt /tmp/requirements-rl.sophia.txt
fi
if [ "$SOPHIA_QUANT" = "bf16" ]; then
  grep -v '^bitsandbytes' /tmp/requirements-rl.sophia.txt > /tmp/requirements-rl.sophia.no-bnb.txt
  mv /tmp/requirements-rl.sophia.no-bnb.txt /tmp/requirements-rl.sophia.txt
fi
python - <<'PY'
from pathlib import Path
p = Path("/tmp/requirements-rl.sophia.txt")
pins = {
    # Coherent vLLM-colocate stack. trl 0.19 has the vllm_mode selector (colocate is
    # the DEFAULT: vLLM in-process on one GPU, no server). vllm 0.9.1 is the matching
    # vLLM API. transformers MUST be pinned to the 4.53.x line: vllm 0.9.1 registers
    # an `aimv2` AutoConfig that already exists in transformers >=5.x, so an unpinned
    # transformers (pip picks 5.x) crashes GRPOTrainer import. datasets pinned off the
    # 5.x line for the same float-too-new reason. peft/accelerate resolve compatibly.
    "trl": "trl==0.19.1",
    "vllm": "vllm==0.9.1",
    "transformers": "transformers==4.53.2",
    "datasets": "datasets==3.6.0",
}
out = []
for line in p.read_text().splitlines():
    stripped = line.strip()
    replaced = False
    for name, pin in pins.items():
        if stripped == name or stripped.startswith(name + ">") or stripped.startswith(name + "="):
            out.append(pin)
            replaced = True
            break
    if not replaced:
        out.append(line)
p.write_text("\\n".join(out) + "\\n")
print("Pinned RLVR requirements:")
print(p.read_text())
PY
"""

    live_cmd = ""
    if args.remote_mode == "live":
        # --capability-panel threads through to eval_rlvr_adapter.py (provenance task).
        # When set, the capability-delta panel runs as part of the held-out eval and is
        # embedded in the .adapter-eval.json (additive; legacy numbers unchanged). The
        # leading space (or empty string) keeps the bash line well-formed either way.
        capability_flag = " --capability-panel" if getattr(args, "capability_panel", False) else ""
        live_cmd = f"""
# vLLM colocate/server runs through vLLM's external-launcher executor, which reads
# the distributed env (RANK/WORLD_SIZE/LOCAL_RANK) that `accelerate launch` sets.
# Plain `python` leaves RANK unset -> KeyError: 'RANK'. --vllm none has no such
# executor and runs fine under plain python.
if [ "$SOPHIA_VLLM" = "none" ]; then
  SOPHIA_LAUNCH="python"
else
  SOPHIA_LAUNCH="accelerate launch --num_processes 1 --num_machines 1"
fi
# Faithfulness-only flags: live entailment provider + a case cap for cheap validation.
# Empty vars expand to nothing (no-op for the other tasks).
ENT_FLAG=""; [ -n "$SOPHIA_ENTAILMENT" ] && ENT_FLAG="--entailment-provider $SOPHIA_ENTAILMENT"
LIMIT_FLAG=""; [ "$SOPHIA_LIMIT" != "0" ] && LIMIT_FLAG="--limit $SOPHIA_LIMIT"
$SOPHIA_LAUNCH tools/run_rlvr.py \\
  --task "$SOPHIA_TASK" \\
  --step-domain "$SOPHIA_STEP_DOMAIN" \\
  --reward "$SOPHIA_REWARD" \\
  --model "$SOPHIA_MODEL" \\
  --quant "$SOPHIA_QUANT" \\
  --vllm "$SOPHIA_VLLM" \\
  --epochs "$SOPHIA_EPOCHS" \\
  --seed "$SOPHIA_SEED" \\
  --out /workspace/sophia-runpod/rlvr.public-report.json \\
  --output /workspace/sophia-runpod/checkpoints/sophia-rlvr-v1 \\
  $ENT_FLAG $LIMIT_FLAG
if [ -d /workspace/sophia-runpod/checkpoints/sophia-rlvr-v1 ]; then
  tar -czf /workspace/sophia-runpod/sophia-rlvr-v1.tar.gz \\
    -C /workspace/sophia-runpod/checkpoints sophia-rlvr-v1
  sha256sum /workspace/sophia-runpod/sophia-rlvr-v1.tar.gz \\
    | tee /workspace/sophia-runpod/sophia-rlvr-v1.tar.gz.sha256
  # Produce the held-out before/after adapter-eval the SSIL Layer-1 gate ingests.
  # Non-fatal: if eval OOMs or fails, the training artifacts are still copied back.
  # --capability-panel (provenance task) runs the capability-delta panel as part of
  # this eval and embeds it in the .adapter-eval.json (additive; legacy unchanged).
  # The faithfulness task is NOT scored by eval_rlvr_adapter (provenance/code-specific);
  # its base-vs-adapter held-out eval (faithfulness_eval with the trained adapter AS the
  # local-HF policy) is a separate step — Open in the ledger. Here we only run the
  # offline instrument smoke so a bad faithfulness build fails loudly on the pod.
  if [ "$SOPHIA_TASK" = "faithfulness" ]; then
    python tools/eval_faithfulness.py --mock \\
      || echo "[runpod] faithfulness instrument smoke failed (non-fatal)"
    # Base-vs-adapter held-out faithfulness contrast on the TRAINED adapter (local-HF
    # policy seam). Entailment = the training verifier if set, else the lexical placeholder.
    EVAL_ENT="${{SOPHIA_ENTAILMENT:-lexical}}"
    EVAL_LIM=""; [ "$SOPHIA_LIMIT" != "0" ] && EVAL_LIM="--limit $SOPHIA_LIMIT"
    python tools/eval_faithfulness.py --compare \\
      --policy "hf:$SOPHIA_MODEL" \\
      --adapter /workspace/sophia-runpod/checkpoints/sophia-rlvr-v1 \\
      --entailment "$EVAL_ENT" $EVAL_LIM \\
      --out /workspace/sophia-runpod/sophia-faithful-v1.compare-eval.json \\
      || echo "[runpod] faithfulness compare-eval failed (non-fatal)"
  else
    python tools/eval_rlvr_adapter.py --mode real \\
      --task "$SOPHIA_TASK" \\
      --step-domain "$SOPHIA_STEP_DOMAIN" \\
      --model "$SOPHIA_MODEL" \\
      --adapter /workspace/sophia-runpod/checkpoints/sophia-rlvr-v1 \\
      --seed "$SOPHIA_SEED" \\
      --out /workspace/sophia-runpod/sophia-rlvr-v1.adapter-eval.json{capability_flag} \\
      || echo "[runpod] adapter-eval failed (non-fatal); no SSIL gate input produced"
  fi
  # PRIMARY metric for the code task: the POWERED N=175 open-invention suite
  # (compositional generalization), scored by the guarded grader. The eval above is
  # the SECONDARY coarse 48-task lane. Both run on the SAME trained adapter; the
  # invention report carries the integrity gate (checks.noRewardHacksAccepted) and
  # the power flag (checks.powered) the measurement_spec treats as primary.
  if [ "$SOPHIA_TASK" = "code" ]; then
    python tools/eval_rlvr_adapter.py --mode real \\
      --task invention \\
      --model "$SOPHIA_MODEL" \\
      --adapter /workspace/sophia-runpod/checkpoints/sophia-rlvr-v1 \\
      --seed "$SOPHIA_SEED" \\
      --out /workspace/sophia-runpod/sophia-rlvr-v1.invention-eval.json \\
      || echo "[runpod] invention-eval failed (non-fatal)"
  fi
fi
"""
    else:
        live_cmd = """
cp /workspace/sophia-runpod/rlvr.offline-report.json /workspace/sophia-runpod/rlvr.public-report.json
echo "Sophia RunPod remote-mode=offline smoke test complete; skipped live GRPO."
"""

    return f"""
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/.cache/huggingface
export TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers
export PIP_CACHE_DIR=/workspace/.cache/pip
export SOPHIA_MODEL={shlex.quote(args.model)}
export SOPHIA_TASK={shlex.quote(args.task)}
export SOPHIA_STEP_DOMAIN={shlex.quote(args.step_domain)}
export SOPHIA_REWARD={shlex.quote(args.reward)}
export SOPHIA_ENTAILMENT={shlex.quote(getattr(args, "entailment_provider", ""))}
export SOPHIA_LIMIT={shlex.quote(str(getattr(args, "limit", 0)))}
export SOPHIA_QUANT={shlex.quote(args.quant)}
export SOPHIA_VLLM={shlex.quote(args.vllm)}
export SOPHIA_EPOCHS={shlex.quote(str(args.epochs))}
export SOPHIA_SEED={shlex.quote(str(args.seed))}
mkdir -p /workspace/sophia-runpod /workspace/.cache/huggingface /workspace/.cache/pip
cd /workspace/sophia-runpod
if [ {shlex.quote(args.source)} = "git" ] && [ ! -d sophia-agi/.git ]; then
  git clone --depth 1{(" --branch " + shlex.quote(args.branch)) if args.branch else ""} {shlex.quote(args.repo_url)} sophia-agi
fi
cd sophia-agi
(git rev-parse HEAD || true) | tee /workspace/sophia-runpod/repo-head.txt
python - <<'PY'
from pathlib import Path
p = Path("tools/run_rlvr.py")
text = p.read_text()
old = 'GLM_TARGET_MODULES = ["query_key_value", "dense", "dense_h_to_4h", "dense_4to_h"]'
new = 'GLM_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]'
if old in text:
    p.write_text(text.replace(old, new))
    print("patched GLM_TARGET_MODULES for current HF GLM module names")
else:
    print("GLM_TARGET_MODULES already current or pattern not found")
PY
nvidia-smi || true
python - <<'PY'
try:
    import torch
    print("torch preinstalled:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
except Exception as exc:
    print("torch precheck failed:", type(exc).__name__, exc)
PY
python -m pip install --upgrade pip setuptools wheel
# The math task's reward is sympy (math_equivalent); needed in BOTH modes so the
# offline smoke's math invariants and the live reward can run.
SOPHIA_NEED_SYMPY=0
if [ "$SOPHIA_TASK" = "math" ]; then SOPHIA_NEED_SYMPY=1; fi
if [ "$SOPHIA_TASK" = "step" ] && [ "$SOPHIA_STEP_DOMAIN" = "math" ]; then SOPHIA_NEED_SYMPY=1; fi
if [ "$SOPHIA_NEED_SYMPY" = "1" ]; then
  python -m pip install -r requirements-math.txt
fi
# The code task's reward executes model-generated code (provenance_bench.code_exec);
# opt in on this ephemeral GPU pod (the executor is time-boxed + process-group-isolated).
# Needed for BOTH the offline smoke (code_reward.offline_invariants) and the live reward.
if [ "$SOPHIA_TASK" = "code" ]; then
  export SOPHIA_ALLOW_CODE_EXEC=1
fi
if [ {shlex.quote(args.remote_mode)} = "live" ]; then
{req_cmd}
  python -m pip install -r /tmp/requirements-rl.sophia.txt
fi
python tools/validate_attribution.py
python tools/run_rlvr.py --task "$SOPHIA_TASK" --step-domain "$SOPHIA_STEP_DOMAIN" --model mock --dry-run --out /workspace/sophia-runpod/rlvr.offline-report.json
{live_cmd}
echo "Sophia RLVR remote run complete."
ls -lh /workspace/sophia-runpod/rlvr*.json || true
"""


def _scp_from_pod(conn: PodConnection, key_path: Path, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "scp",
        "-i",
        str(key_path),
        "-P",
        str(conn.ssh_port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{conn.public_ip}:{remote}",
        str(local),
    ]
    proc = _run(cmd, check=False)
    return proc.returncode == 0


def _rsync_repo_to_pod(conn: PodConnection, key_path: Path) -> None:
    ssh_cmd = (
        "ssh "
        + " ".join(
            shlex.quote(x)
            for x in [
                "-i",
                str(key_path),
                "-p",
                str(conn.ssh_port),
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
            ]
        )
    )
    remote = f"root@{conn.public_ip}:/workspace/sophia-runpod/sophia-agi/"
    cmd = [
        "rsync",
        "-rltz",
        "--delete",
        "--no-owner",
        "--no-group",
        "--no-perms",
        "--exclude",
        ".git/",
        "--exclude",
        ".venv*/",
        "--exclude",
        "__pycache__/",
        "--exclude",
        ".pytest_cache/",
        "--exclude",
        "training/rlvr/checkpoints/",
        "-e",
        ssh_cmd,
        str(ROOT) + "/",
        remote,
    ]
    _run(_ssh_base(conn, key_path) + ["mkdir -p /workspace/sophia-runpod/sophia-agi"])
    _run(cmd)


def _delete_pod(api_key: str, pod_id: str) -> None:
    print(f"[runpod] terminating pod {pod_id} ...")
    try:
        _api_request("DELETE", f"/pods/{pod_id}", api_key, timeout=60)
        print(f"[runpod] pod {pod_id} terminated")
    except RunPodError as exc:
        # A 404 can happen if the Pod already self-exited/was deleted manually.
        print(f"[runpod] WARNING: delete failed for pod {pod_id}: {exc}")


def _find_pod_by_name(api_key: str, name: str) -> dict[str, Any] | None:
    pods = _api_request("GET", "/pods", api_key, timeout=60)
    for pod in pods or []:
        if pod.get("name") == name and pod.get("desiredStatus") != "TERMINATED":
            return pod
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key-env", default="RUNPOD_API_KEY", help="environment variable containing the RunPod API key")
    ap.add_argument(
        "--api-key-file",
        type=Path,
        default=None,
        help="optional file containing the RunPod API key; useful for /private/tmp secrets outside the repo",
    )
    ap.add_argument("--yes", action="store_true", help="actually create a RunPod pod (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="print sanitized payload and remote command without creating a pod")
    ap.add_argument("--keep-pod", action="store_true", help="do NOT delete the pod after the run; use only for debugging")
    ap.add_argument(
        "--local",
        action="store_true",
        help="run RLVR on the LOCAL GPU (e.g. NVIDIA DGX Spark) instead of renting a RunPod pod; "
             "no SSH/pod/cost. For aarch64/Grace Blackwell pair with --quant bf16 --vllm none to "
             "sidestep the flash-attn/bitsandbytes/vLLM-colocate/unsloth wheel blockers.",
    )
    ap.add_argument("--name", default=f"sophia-rlvr-{timestamp}")
    ap.add_argument("--source", choices=["local", "git"], default="local", help="upload current working tree or clone GitHub")
    ap.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    ap.add_argument("--branch", default="", help="optional git branch/tag to clone")
    ap.add_argument("--model", default="zai-org/glm-4-9b-chat-hf")
    ap.add_argument("--remote-mode", choices=["offline", "live"], default="live", help="run only remote offline smoke test or full live GRPO")
    ap.add_argument("--task", choices=["provenance", "math", "code", "concept", "step", "faithfulness"], default="provenance", help="RLVR reward task: provenance (provenance_faithful), math (sympy math_equivalent), code (hidden-tests-pass via code_exec), concept (concept-TBox gate), step (process: every step verified), or faithfulness (retrieve-then-reason GRPO + counterfactual citation-drop)")
    ap.add_argument("--step-domain", choices=["math", "physics"], default="math", help="for --task step: per-step oracle + held-out RL split (math needs sympy; physics is pure-Python)")
    ap.add_argument("--reward", choices=["verifier", "gate", "multiaxis"], default="verifier", help="reward signal (provenance task): verifier (default), gate (single-axis), or multiaxis (Thesis D dense reward; M1 collapse comparison)")
    ap.add_argument("--entailment-provider", choices=["", "deepseek", "llmhub"], default="",
                    help="faithfulness task: live entailment LLM behind the verify seam "
                         "(needs DEEPSEEK_API_KEY / LLMHUB_API_KEY forwarded to the pod); "
                         "blank = lexical placeholder")
    ap.add_argument("--limit", type=int, default=0,
                    help="faithfulness task: cap training cases (0 = all; use e.g. 24 for a cheap validation)")
    ap.add_argument("--quant", choices=["bf16", "4bit"], default="bf16")
    ap.add_argument("--vllm", choices=["none", "server", "colocate"], default="none")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0, help="training+eval seed; vary across runs for independent replications")
    ap.add_argument("--capability-panel", action="store_true",
                    help="also run the capability-delta panel (attribution/hallucination/calibration) "
                         "as part of the held-out adapter eval and embed it in the .adapter-eval.json. "
                         "Provenance task only; additive evidence (legacy numbers unchanged).")
    ap.add_argument("--gpu-type", default=",".join(DEFAULT_GPU_TYPES), help="comma-separated RunPod GPU type preference list")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--cloud-type", choices=["SECURE", "COMMUNITY"], default="SECURE")
    ap.add_argument("--interruptible", action="store_true", help="use spot/interruptible pod")
    ap.add_argument("--image-name", default=DEFAULT_IMAGE)
    ap.add_argument("--container-disk-gb", type=int, default=120)
    ap.add_argument("--volume-gb", type=int, default=80)
    ap.add_argument("--network-volume-id", default="",
                    help="attach an existing persistent RunPod network volume at /workspace (HF_HOME) so the "
                         "weight cache survives pod deletion; pod is then pinned to that volume's data center. "
                         "Run tools/runpod_volume_breakeven.py first — only pays off past the break-even runs/mo.")
    ap.add_argument("--allowed-cuda-versions", default="")
    ap.add_argument("--ssh-timeout-s", type=int, default=1200)
    ap.add_argument("--auto-exit-seconds", type=int, default=6 * 60 * 60)
    ap.add_argument(
        "--no-remote-delete-watchdog",
        action="store_true",
        help="do not pass the RunPod API key into the Pod for the max-runtime delete watchdog",
    )
    ap.add_argument("--artifacts-dir", type=Path, default=ROOT / "agi-proof" / "benchmark-results" / "runpod-rlvr")
    return ap.parse_args(argv)


def _run_local(args: argparse.Namespace) -> int:
    """Run RLVR on the LOCAL GPU (e.g. NVIDIA DGX Spark) instead of renting a RunPod pod.

    Short-circuits pod creation / SSH / scp / RunPod cost entirely: the same
    ``tools/run_rlvr.py`` the remote script would invoke is run against the host
    Python+CUDA stack. For aarch64 / Grace Blackwell use ``--quant bf16 --vllm none``
    to avoid the flash-attn / bitsandbytes / vLLM-colocate / unsloth aarch64-wheel
    blockers (the 128 GB unified memory makes bf16 feasible where rented pods needed
    4-bit). No RunPod API key is required in this mode.
    """
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.artifacts_dir / "local.rlvr.public-report.json"
    cmd = [
        sys.executable, str(ROOT / "tools" / "run_rlvr.py"),
        "--task", args.task, "--reward", args.reward, "--model", args.model, "--quant", args.quant,
        "--vllm", args.vllm, "--epochs", str(args.epochs), "--seed", str(args.seed),
        "--out", str(out_path),
    ]
    if args.remote_mode == "offline":
        # run_rlvr's own --dry-run is the offline reward-wiring check (no GPU, no model load).
        cmd.append("--dry-run")
    print("[runpod] --local: running RLVR on the LOCAL GPU (no pod, no SSH, no RunPod cost)")
    print("[runpod] local command: " + " ".join(cmd))
    if args.dry_run:
        print("[runpod] dry-run only; nothing executed")
        return 0
    if not args.yes:
        raise RunPodError("Refusing to run RLVR locally without --yes. Use --dry-run to inspect the command first.")
    log_path = args.artifacts_dir / "local.train.log"
    exit_code = _stream(cmd, log_path)
    print(f"[runpod] local command exit code: {exit_code}; log={log_path}; report={out_path}")
    return exit_code


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.local:
        return _run_local(args)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and args.api_key_file:
        api_key = args.api_key_file.read_text(encoding="utf-8").strip()

    if not shutil.which("ssh"):
        raise RunPodError("ssh not found on PATH")
    if not shutil.which("scp"):
        raise RunPodError("scp not found on PATH")
    if not shutil.which("ssh-keygen"):
        raise RunPodError("ssh-keygen not found on PATH")

    with tempfile.TemporaryDirectory(prefix="sophia-runpod-") as tmp:
        tmpdir = Path(tmp)
        key_path, public_key = _generate_ssh_key(tmpdir)
        payload = _build_create_payload(args, public_key, api_key=api_key)
        sanitized_payload = json.loads(json.dumps(payload))
        sanitized_payload["env"]["PUBLIC_KEY"] = "ssh-ed25519 …"
        if "RUNPOD_API_KEY" in sanitized_payload["env"]:
            sanitized_payload["env"]["RUNPOD_API_KEY"] = _redact(api_key)

        print(f"[runpod] api key env={args.api_key_env}, value={_redact(api_key)}")
        print("[runpod] create payload (sanitized):")
        print(json.dumps(sanitized_payload, indent=2))
        print("[runpod] remote training script:")
        print(_remote_training_script(args))

        if args.dry_run:
            print("[runpod] dry-run only; no pod created")
            return 0
        if not args.yes:
            raise RunPodError("Refusing to create a paid RunPod pod without --yes. Use --dry-run to inspect first.")
        if not api_key:
            raise RunPodError(f"Set {args.api_key_env}=<RunPod API key> before running.")

        pod_id = ""
        conn: PodConnection | None = None
        exit_code = 1
        try:
            try:
                pod = _api_request("POST", "/pods", api_key, payload)
            except RunPodError as exc:
                # RunPod can occasionally return a 5xx after the Pod was
                # actually created. Recover by finding our unique Pod name so
                # the normal cleanup path still owns the billable resource.
                print(f"[runpod] create returned an error; scanning for orphan named {args.name!r}: {exc}")
                pod = _find_pod_by_name(api_key, args.name)
                if not pod:
                    raise
                print(f"[runpod] recovered created pod after error: {pod.get('id')}")
            pod_id = _pod_id(pod)
            print(f"[runpod] created pod {pod_id}; costPerHr={pod.get('costPerHr')}, gpu={pod.get('gpu')}")
            conn = _poll_ssh(api_key, pod_id, timeout_s=args.ssh_timeout_s)
            print(f"[runpod] SSH mapped: root@{conn.public_ip} -p {conn.ssh_port}")
            _wait_ssh_login(conn, key_path)
            if args.source == "local":
                _rsync_repo_to_pod(conn, key_path)

            args.artifacts_dir.mkdir(parents=True, exist_ok=True)
            log_path = args.artifacts_dir / f"{pod_id}.train.log"
            remote_script = _remote_training_script(args)
            cmd = _ssh_base(conn, key_path) + ["bash", "-s"]
            exit_code = _stream(cmd, log_path, input_text=remote_script)
            print(f"[runpod] remote command exit code: {exit_code}; log={log_path}")

            _scp_from_pod(
                conn,
                key_path,
                "/workspace/sophia-runpod/rlvr.public-report.json",
                args.artifacts_dir / f"{pod_id}.rlvr.public-report.json",
            )
            _scp_from_pod(
                conn,
                key_path,
                "/workspace/sophia-runpod/rlvr.offline-report.json",
                args.artifacts_dir / f"{pod_id}.rlvr.offline-report.json",
            )
            # Before/after adapter-eval (live mode only; best-effort). The workflow's
            # ingest step globs *adapter-eval*.json and runs it through the SSIL gate.
            _scp_from_pod(
                conn,
                key_path,
                "/workspace/sophia-runpod/sophia-rlvr-v1.adapter-eval.json",
                args.artifacts_dir / f"{pod_id}.rlvr.adapter-eval.json",
            )
            # POWERED open-invention primary eval (code task; best-effort).
            _scp_from_pod(
                conn,
                key_path,
                "/workspace/sophia-runpod/sophia-rlvr-v1.invention-eval.json",
                args.artifacts_dir / f"{pod_id}.rlvr.invention-eval.json",
            )
            # Base-vs-adapter faithfulness contrast (faithfulness task; best-effort).
            _scp_from_pod(
                conn,
                key_path,
                "/workspace/sophia-runpod/sophia-faithful-v1.compare-eval.json",
                args.artifacts_dir / f"{pod_id}.faithful.compare-eval.json",
            )
            _scp_from_pod(
                conn,
                key_path,
                "/workspace/sophia-runpod/repo-head.txt",
                args.artifacts_dir / f"{pod_id}.repo-head.txt",
            )
            _scp_from_pod(
                conn,
                key_path,
                "/workspace/sophia-runpod/sophia-rlvr-v1.tar.gz",
                args.artifacts_dir / f"{pod_id}.sophia-rlvr-v1.tar.gz",
            )
            _scp_from_pod(
                conn,
                key_path,
                "/workspace/sophia-runpod/sophia-rlvr-v1.tar.gz.sha256",
                args.artifacts_dir / f"{pod_id}.sophia-rlvr-v1.tar.gz.sha256",
            )
            return exit_code
        finally:
            if pod_id and not args.keep_pod:
                _delete_pod(api_key, pod_id)
            elif pod_id:
                print(f"[runpod] --keep-pod set; pod still running: {pod_id}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunPodError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

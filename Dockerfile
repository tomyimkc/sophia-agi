# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
#
# Prebaked CUDA training image for the Sophia LoRA / RLVR RunPod runs.
#
# WHY THIS EXISTS (cold-start = the biggest measured wall-clock waste):
#   On a vanilla runpod/pytorch image, every pod re-ran `pip install -r
#   requirements-lora.txt` AND compiled flash-attn FROM SOURCE. The flash-attn
#   compile alone dominated our runs (~hours of paid GPU time, repeated per pod).
#   Baking a PINNED, PREBUILT flash-attn wheel + the LoRA deps into the image
#   turns that into a one-time build so live pods short-circuit the slow installs
#   (tools/runpod_speedup.py guards the pip steps with an "already importable?"
#   check, so this image makes them no-ops).
#
# BUILD & PUSH (maintainer, one-time per dependency bump):
#   docker build -t <dockerhub-user>/sophia-train:latest .
#   docker push <dockerhub-user>/sophia-train:latest
# Then launch with the prebaked image, e.g.:
#   python tools/runpod_speedup.py --dry-run \
#       --image-name <dockerhub-user>/sophia-train:latest
#   (or set the workflow_dispatch 'image' input to the same tag).
#
# Base pinned to torch 2.8.0 (NOT the torch-2.9.1 RLVR default): torch 2.9 has NO
# prebuilt flash-attn wheel yet, so it would compile from source (~the 90-min timeout
# we hit). torch 2.8 has an official cu12 wheel — see DEFAULT_BENCH_IMAGE in
# tools/runpod_speedup.py. Keep this base and that constant in sync.
FROM runpod/pytorch:1.0.7-cu1281-torch280-ubuntu2204

# Persistent HF cache path (mirrors the remote bench script export so a cached
# model resolves to the same location whether or not the volume is mounted).
ENV HF_HOME=/workspace/.cache/huggingface \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Pin the LoRA dependency closure into the image. Copy ONLY the requirements file
# first so this layer caches independently of the rest of the tree.
COPY requirements-lora.txt /opt/sophia/requirements-lora.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /opt/sophia/requirements-lora.txt && \
    # Install the PINNED PREBUILT flash-attn wheel (downloads in seconds, NO compile),
    # auto-detecting the cp tag + cxx11 ABI from the image's python/torch.
    python - <<'PY' && \
import subprocess, sys
import torch
mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
cp = f"cp{sys.version_info.major}{sys.version_info.minor}"
abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
url = (f"https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/"
       f"flash_attn-2.8.3.post1+cu12torch{mm}cxx11abi{abi}-{cp}-{cp}-linux_x86_64.whl")
print("installing", url, flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", url])
PY
    # Unsloth fused-kernel backend (the 4-bit config in the speedup bench).
    python -m pip install "unsloth>=2024.8" && \
    # Fail the BUILD (not the live pod) if either heavy import is broken, so a bad
    # image never ships and silently falls back to the slow runtime install path.
    python -c "import flash_attn, unsloth; print('prebaked flash_attn', flash_attn.__version__)" && \
    rm -rf /root/.cache/pip

WORKDIR /workspace

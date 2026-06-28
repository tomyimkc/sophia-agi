# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Device/tier resolver for the from-scratch GPT, encoding the cluster rules.

The dev cluster is heterogeneous and the charter assigns each node a *role*, not
just a device string (see ``docs/11-Platform/DGX-Spark.md`` and
``docs/06-Roadmap/Sophia-Wisdom-4B-M3-Pilot.md``):

  - **DGX Spark** (GB10, aarch64, 128 GB unified) → CUDA, **bf16** (not 4-bit;
    bitsandbytes aarch64 is painful), **never MLX**. Iteration tier.
  - **Mac Studio M3 Ultra** (96 GB unified) → Apple Silicon: torch **MPS** here,
    or hand off to MLX tooling (``tools/eval_mlx_model.py``). Apple tier.
  - **CPU** → always-correct fallback; CI runs here.

The honest boundary this resolver re-states: a Spark/M3 run is **iteration**, and
``canClaimAGI`` stays false. Headline numbers stay on x86 RunPod.

This module imports torch lazily so the package stays importable without it.
"""
from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceTier:
    device: str          # "cuda" | "mps" | "cpu"
    dtype: str           # "bfloat16" | "float16" | "float32"
    tier: str            # "spark" | "m3" | "cuda" | "cpu"
    headline_ok: bool    # may this node's numbers be cited as registered evidence?
    note: str

    def as_dict(self) -> dict:
        return {
            "device": self.device, "dtype": self.dtype, "tier": self.tier,
            "headline_ok": self.headline_ok, "note": self.note,
        }


def _is_aarch64() -> bool:
    return platform.machine().lower() in {"aarch64", "arm64"}


def resolve_tier(prefer: str = "auto") -> DeviceTier:
    """Pick a device + dtype + role. ``prefer`` may be auto/cuda/mps/cpu.

    Falls back to CPU whenever torch is missing or the requested accelerator is
    unavailable — never raises, so the same script runs on every node and in CI.
    """
    try:
        import torch  # noqa: PLC0415 — optional, lazy
    except Exception:  # noqa: BLE001 — torch optional
        return DeviceTier("cpu", "float32", "cpu", False,
                          "torch absent — tokenizer/research path only")

    want = prefer.lower()
    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if want in {"auto", "cuda"} and cuda:
        # aarch64 + CUDA is the DGX Spark signature.
        if _is_aarch64():
            return DeviceTier("cuda", "bfloat16", "spark", False,
                              "DGX Spark (GB10) — iteration tier, bf16, never MLX; "
                              "headline stays x86 RunPod")
        return DeviceTier("cuda", "bfloat16", "cuda", False,
                          "x86 CUDA — set headline_ok only for the registered RunPod lane")
    if want in {"auto", "mps"} and mps:
        # torch MPS is fp32/fp16; bf16 on MPS is uneven, so default fp16.
        return DeviceTier("mps", "float16", "m3", False,
                          "Mac Studio M3 — Apple tier; for LoRA/serve prefer MLX tooling")
    return DeviceTier("cpu", "float32", "cpu", False, "CPU fallback (CI / any machine)")


def torch_dtype(tier: DeviceTier):
    """Map the tier's dtype name to a torch dtype (lazy import)."""
    import torch  # noqa: PLC0415
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[tier.dtype]


__all__ = ["DeviceTier", "resolve_tier", "torch_dtype"]

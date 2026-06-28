# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Pretraining-architecture reference studies (nano scale, pure Python + numpy).

Nano-scale studies of sparse / routed architectures against a known entropy
floor. Each submodule is a numpy / pure-Python reference CI-checked for its
core policy via ``offline_invariants()``; none is a GPU deployment or a SOTA
claim. See ``pretraining/architecture/ARCHITECTURE.md`` for the real
DeepSeek MLA + fine-grained-MoE design these toys gesture at.

Submodules:

- ``moe``           : ``MoELM`` — a minimal top-1 MoE hidden layer (trainable).
- ``run_arch``      : dense-vs-MoE probe at matched active compute.
- ``run_sparse_quant`` : sparsity + adaptive-quant composition against the floor.
- ``p7_router_ablation`` : P7 — router-policy ablation on fixed MoE experts.
- ``recurrent_depth`` : RDT — the looped-transformer mechanism (OpenMythos / Mythos
  reconstruction) at nano scale: LTI stability, depth extrapolation, weight sharing.
"""
from __future__ import annotations

from pretraining.architecture.moe import MoELM
from pretraining.architecture.p7_router_ablation import (
    SCOPE_KEY,
    offline_invariants as p7_offline_invariants,
    run_ablation as p7_run_ablation,
)
from pretraining.architecture.recurrent_depth import (
    NanoRDT,
    offline_invariants as rdt_offline_invariants,
    run_study as rdt_run_study,
)
from pretraining.architecture.run_arch import run as run_arch_probe
from pretraining.architecture.run_sparse_quant import run as run_sparse_quant

__all__ = [
    "MoELM",
    # P7 router-policy ablation
    "SCOPE_KEY",
    "p7_offline_invariants",
    "p7_run_ablation",
    # Recurrent-Depth Transformer (looped-transformer mechanism)
    "NanoRDT",
    "rdt_offline_invariants",
    "rdt_run_study",
    # study entry points
    "run_arch_probe",
    "run_sparse_quant",
]

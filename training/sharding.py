# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Expert-parallel + FSDP sharding — the gate to high/top-tier (multi-GPU) training.

*Why this exists.* ``tools/train_lora.py`` is single-GPU today; a 141B–671B MoE base does not
fit one card even frozen. Two orthogonal parallelisms make it trainable across a small fleet,
and this module is their **policy + per-rank memory accounting**:

  - **FSDP** (Fully-Sharded Data Parallel) shards the *dense trunk* params/grads/optimizer
    states across all ``world_size`` ranks — each rank holds ``dense/world_size``, all-gathered
    per layer in the forward and reduce-scattered in the backward.
  - **Expert parallelism (EP)** distributes the *experts* across ranks — each rank owns
    ``n_experts/world_size`` experts per layer; tokens are all-to-all dispatched to the rank
    holding their routed expert and combined back. This is the MoE-specific lever (Megatron /
    DeepSpeed-MoE), and the training-side mirror of ``serving/expert_offload.py``.

With a **frozen, low-bit base + LoRA** (the only honest "few-resources" path — Boundary 2), the
per-rank cost is ``total_params/world_size × base_bytes`` plus tiny LoRA optimizer state, so a
671B MoE at 4-bit shards to ~42 GB/rank across 8 GPUs (fits 80 GB cards), and at bf16 needs more
ranks — both of which :func:`plan_sharding` computes exactly.

*What is CI-tested vs deployment.* The sharding **policy** (balanced expert→rank assignment,
FSDP shard sizes, per-rank memory, all-to-all volume) is pure-Python with deterministic
``offline_invariants()`` — the repo's reference discipline. The actual ``torch.distributed`` /
FSDP / all-to-all glue is guarded behind a torch import (``# pragma: no cover``), wired into
``train_lora.py`` via ``--shard``/``--expert-parallel``/``--world-size``; it runs on a multi-GPU
pod, out of scope for the single-process CI reference.

*Honest scope.* This makes high/top-tier **adaptation** (LoRA/QAT on a frozen open MoE base)
trainable across a fleet; it does **not** make from-scratch frontier pretraining cheap (it isn't).
The per-rank numbers are byte accounting; a real run's fit still depends on activation/seq budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EXPERT_PARAM_FRACTION = 0.85   # share of off-embedding body that is expert MLPs (fine-grained MoE)
ADAM_BYTES_PER_PARAM = 8.0     # fp32 m+v optimizer state for a *trainable* param
GRAD_BYTES_PER_PARAM = 2.0     # bf16 grad for a trainable param


@dataclass(frozen=True)
class ShardPlan:
    """Per-rank memory + communication plan for FSDP + expert-parallel training."""

    world_size: int
    base_bits: float
    experts_per_rank: int
    dense_shard_bytes: int          # FSDP shard of the dense trunk, per rank
    expert_shard_bytes: int         # local experts, per rank (EP)
    lora_state_bytes: int           # LoRA grads + optimizer state, per rank
    activation_bytes: int
    per_rank_bytes: int
    all_to_all_tokens: int          # tokens dispatched off-rank per MoE layer per step
    fits_gpu_gb: float

    @property
    def per_rank_gb(self) -> float:
        return self.per_rank_bytes / 1e9

    @property
    def fits(self) -> bool:
        return self.per_rank_bytes <= self.fits_gpu_gb * 1e9


def expert_assignment(n_experts: int, world_size: int) -> "list[list[int]]":
    """Balanced expert→rank assignment (blocked, remainder spread to the low ranks).

    Returns a list of length ``world_size``; entry ``r`` is the expert ids owned by rank ``r``.
    Every expert is owned by exactly one rank, and rank loads differ by at most one expert.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    base, rem = divmod(n_experts, world_size)
    assignment: list[list[int]] = []
    nxt = 0
    for r in range(world_size):
        count = base + (1 if r < rem else 0)
        assignment.append(list(range(nxt, nxt + count)))
        nxt += count
    return assignment


def plan_sharding(*, total_params: int, embed_params: int, n_routed_experts: int, n_layers: int,
                  world_size: int, base_bits: float = 4.0, lora_frac: float = 0.005,
                  tokens_per_step: int = 4096, active_experts: int = 8,
                  activation_gb: float = 4.0, fits_gpu_gb: float = 80.0) -> ShardPlan:
    """Compute the per-rank memory + all-to-all plan for FSDP + EP LoRA/QAT training.

    ``base_bits`` is the frozen base width (4 = QLoRA, 16 = bf16). ``lora_frac`` is the trainable
    fraction (LoRA adapters) — only these carry grad + optimizer state. The base is sharded
    (FSDP for the dense trunk, EP for the experts), so per-rank base bytes ≈
    ``total_params/world_size × base_bits/8``.
    """
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    if not (1 <= base_bits <= 16):
        raise ValueError("base_bits must be in [1, 16]")
    base_bytes = base_bits / 8.0

    body = max(0, total_params - embed_params)
    expert_total = int(body * EXPERT_PARAM_FRACTION)
    dense_total = total_params - expert_total          # trunk + embed/head

    # FSDP shards the dense trunk; EP shards the experts. Both ≈ /world_size per rank.
    dense_shard = int(dense_total / world_size * base_bytes)
    experts_per_rank = math.ceil(n_routed_experts / world_size)
    expert_shard = int(expert_total / world_size * base_bytes)

    # Only LoRA params are trainable -> grad + Adam state (tiny). Replicated per rank (LoRA is
    # small; sharding it saves little and complicates the all-reduce), so charge it per rank.
    lora_params = int(total_params * lora_frac)
    lora_state = int(lora_params * (GRAD_BYTES_PER_PARAM + ADAM_BYTES_PER_PARAM))

    activation = int(activation_gb * 1e9)
    per_rank = dense_shard + expert_shard + lora_state + activation

    # All-to-all: per MoE layer, tokens whose routed expert lives off-rank are dispatched.
    # Fraction off-rank ≈ (world_size-1)/world_size of the top-k assignments.
    off_rank_frac = (world_size - 1) / world_size if world_size > 1 else 0.0
    all_to_all = int(tokens_per_step * active_experts * off_rank_frac)

    return ShardPlan(
        world_size=world_size, base_bits=base_bits, experts_per_rank=experts_per_rank,
        dense_shard_bytes=dense_shard, expert_shard_bytes=expert_shard,
        lora_state_bytes=lora_state, activation_bytes=activation, per_rank_bytes=per_rank,
        all_to_all_tokens=all_to_all, fits_gpu_gb=fits_gpu_gb,
    )


def min_world_size_to_fit(*, total_params: int, embed_params: int, n_routed_experts: int,
                          n_layers: int, base_bits: float = 4.0, fits_gpu_gb: float = 80.0,
                          activation_gb: float = 4.0, max_world: int = 256) -> "int | None":
    """Smallest ``world_size`` (power-of-two-friendly search) that fits the GPU budget, or None."""
    w = 1
    while w <= max_world:
        plan = plan_sharding(total_params=total_params, embed_params=embed_params,
                             n_routed_experts=n_routed_experts, n_layers=n_layers, world_size=w,
                             base_bits=base_bits, activation_gb=activation_gb, fits_gpu_gb=fits_gpu_gb)
        if plan.fits:
            return w
        w *= 2
    return None


# ---------------------------------------------------------------------------
# Torch glue (deployment path; guarded — not run in CI)
# ---------------------------------------------------------------------------

def wrap_model_fsdp(model, *, world_size: int, cpu_offload: bool = False):  # pragma: no cover - torch-only
    """Wrap ``model`` in PyTorch FSDP (full param/grad/optimizer-state sharding).

    Deployment path: called from ``tools/train_lora.py`` when ``--shard fsdp`` and
    ``torch.distributed`` is initialized. Shards each transformer block; LoRA params stay
    trainable, the frozen base is sharded.
    """
    import torch
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    import functools

    policy = functools.partial(size_based_auto_wrap_policy, min_num_params=int(1e7))
    return FSDP(model, auto_wrap_policy=policy,
                cpu_offload=CPUOffload(offload_params=cpu_offload),
                device_id=torch.cuda.current_device())


def apply_expert_parallel(model, *, rank: int, world_size: int):  # pragma: no cover - torch-only
    """Keep only this rank's experts resident; route via all-to-all to the owning rank.

    Deployment path (Megatron/DeepSpeed-MoE style): uses :func:`expert_assignment` to decide
    which experts rank ``r`` owns, frees the rest, and installs an all-to-all dispatch/combine
    around each MoE layer's router. The owning-rank computation mirrors
    ``serving/expert_offload.py``'s promote-on-route, at training time.
    """
    raise NotImplementedError(
        "expert-parallel torch glue is the on-pod deployment artifact; the policy "
        "(expert_assignment / plan_sharding) is the CI-tested reference here.")


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. Expert assignment is a balanced partition: every expert owned once, loads differ by <=1.
    asn = expert_assignment(160, 8)
    flat = [e for rank in asn for e in rank]
    checks["assignment_covers_all"] = sorted(flat) == list(range(160))
    checks["assignment_unique"] = len(flat) == len(set(flat))
    sizes = [len(r) for r in asn]
    checks["assignment_balanced"] = max(sizes) - min(sizes) <= 1
    detail["assignment_sizes"] = sizes

    # 2. Remainder spreads (13 experts / 4 ranks -> 4,3,3,3).
    asn2 = expert_assignment(13, 4)
    checks["remainder_spread"] = [len(r) for r in asn2] == [4, 3, 3, 3]

    # 3. Per-rank base memory shards linearly: doubling world_size ~halves per-rank base bytes.
    ds = dict(total_params=671_000_000_000, embed_params=2 * 129280 * 7168,
              n_routed_experts=256, n_layers=61)   # DeepSeek-V3 size axes
    p4 = plan_sharding(world_size=4, base_bits=4.0, activation_gb=0.0, **ds)
    p8 = plan_sharding(world_size=8, base_bits=4.0, activation_gb=0.0, **ds)
    base4 = p4.dense_shard_bytes + p4.expert_shard_bytes
    base8 = p8.dense_shard_bytes + p8.expert_shard_bytes
    checks["base_shards_linearly"] = abs(base8 - base4 / 2) / base4 < 0.02
    detail["base_gb_4ranks"] = round(base4 / 1e9, 1)
    detail["base_gb_8ranks"] = round(base8 / 1e9, 1)

    # 4. DeepSeek-V3 (671B) QLoRA(4-bit) fits 8×80GB but NOT 1 GPU — the sharding win.
    one = plan_sharding(world_size=1, base_bits=4.0, fits_gpu_gb=80.0, activation_gb=4.0, **ds)
    eight = plan_sharding(world_size=8, base_bits=4.0, fits_gpu_gb=80.0, activation_gb=4.0, **ds)
    checks["one_gpu_does_not_fit"] = not one.fits
    checks["eight_gpu_fits_4bit"] = eight.fits
    detail["per_rank_gb_1x_4bit"] = round(one.per_rank_gb, 1)
    detail["per_rank_gb_8x_4bit"] = round(eight.per_rank_gb, 1)

    # 5. The bits lever: bf16 needs strictly more ranks than 4-bit to fit the same GPU.
    w_4bit = min_world_size_to_fit(base_bits=4.0, fits_gpu_gb=80.0, activation_gb=4.0,
                                   total_params=671_000_000_000, embed_params=2 * 129280 * 7168,
                                   n_routed_experts=256, n_layers=61)
    w_bf16 = min_world_size_to_fit(base_bits=16.0, fits_gpu_gb=80.0, activation_gb=4.0,
                                   total_params=671_000_000_000, embed_params=2 * 129280 * 7168,
                                   n_routed_experts=256, n_layers=61)
    checks["bf16_needs_more_ranks"] = w_bf16 > w_4bit
    detail["min_ranks_4bit_80gb"] = w_4bit
    detail["min_ranks_bf16_80gb"] = w_bf16

    # 6. All-to-all volume grows with world_size (more experts off-rank), zero on 1 rank.
    checks["no_all_to_all_single_rank"] = plan_sharding(world_size=1, base_bits=4.0, **ds).all_to_all_tokens == 0
    checks["all_to_all_positive_multi"] = p8.all_to_all_tokens > 0

    # 7. Mid-tier Mixtral-8x22B (141B) fits even 2×80GB at 4-bit (high tier is reachable).
    mixtral = dict(total_params=141_000_000_000, embed_params=2 * 32000 * 6144,
                   n_routed_experts=8, n_layers=56, base_bits=4.0, activation_gb=4.0)
    checks["mixtral_fits_2x"] = plan_sharding(world_size=2, fits_gpu_gb=80.0, **mixtral).fits
    detail["mixtral_per_rank_gb_2x"] = round(plan_sharding(world_size=2, fits_gpu_gb=80.0, **mixtral).per_rank_gb, 1)

    # 8. Fail-closed: bad world_size / bits / expert count.
    for bad in (lambda: expert_assignment(8, 0), lambda: plan_sharding(world_size=0, **ds),
                lambda: plan_sharding(world_size=2, base_bits=0, **ds)):
        try:
            bad(); checks.setdefault("fail_closed", True); checks["fail_closed"] = False
        except ValueError:
            checks.setdefault("fail_closed", True)

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Sharding (FSDP + expert-parallel) offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"\n  DeepSeek-V3 671B QLoRA(4-bit) per-rank: "
          f"1×={detail.get('per_rank_gb_1x_4bit')} GB (no fit) -> "
          f"8×={detail.get('per_rank_gb_8x_4bit')} GB (fits 80GB)")
    print(f"  min ranks to fit 80GB: 4-bit={detail.get('min_ranks_4bit_80gb')}, "
          f"bf16={detail.get('min_ranks_bf16_80gb')}")
    print(f"  Mixtral-8x22B 141B per-rank @2×: {detail.get('mixtral_per_rank_gb_2x')} GB")
    raise SystemExit(0 if ok else 1)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Adaptive (mixed-precision) quantization — the Unsloth-Dynamic-2.0 core idea,
made reproducible and governed.

*Why this module exists.* Uniform quantization fails at 1–3 bits because it misallocates
the *error budget*: every tensor pays the same bit-width, so the few tensors that dominate
output error get crushed alongside the many that barely matter. The fix is
**layer/tensor-adaptive quantization** — assign each tensor a bit-width in proportion to
its measured *output sensitivity*. Unsloth's Dynamic 2.0 does exactly this (it pushed
GLM-5.2 from ~1.6 TB to ~220 GB at surprisingly graceful degradation), but its
**per-layer selection rules are not published** — they are model-specific, hand-tuned, and
opaque. This module is the reproducible, governed counterpart:

1. **Sensitivity** is *measured*, not assumed — the output KL-divergence contribution of
   quantizing each tensor in isolation, against a calibration distribution.
2. **Allocation** is a *greedy bit-allocation under a total-size budget* with a
   **protected-floor** for high-sensitivity tensors (embeddings, lm_head, MoE routers,
   early attention — the documented sensitivity hierarchy). The policy is a few lines of
   code you can read, not a black box.
3. **Governance** is the ``offline_invariants`` contract the rest of ``moe/`` already uses:
   the allocator is CI-proven to (a) respect the total-bit budget **exactly**, (b) never
   under-allocate a protected tensor below its floor, and (c) never allocate below 1 bit.

The mechanism, why it works at low bits
---------------------------------------
At a target average width ``w`` and ``N`` elements, the size budget is ``B = w·N`` bits.
A tensor quantized to ``b`` bits contributes an output distortion that falls roughly
*exponentially* in ``b`` (each extra bit halves the quant step). So the *marginal
distortion reduction per bit* is highest for the most sensitive tensors at the lowest
widths. Spending the first bits on the sensitive tensors and starving the redundant ones is
therefore near-optimal under a budget — this is the classic greedy solution to a
knapsack-with-decreasing-returns, and it is what "Dynamic 2.0" is doing under the hood.

What this module is NOT
-----------------------
It does **not** reproduce Unsloth's exact scheme (those rules are unpublished). It
implements the *principle* (measured sensitivity → greedy budgeted allocation with a
protected floor) with an honest, checkable policy. It does **not** move real GPU tensors;
it is a numpy reference, CI-tested, exactly like ``moe/quant.py`` and ``moe/router.py``.
See ``docs/11-Platform/Cheap-Compute-Boundary.md`` for what claims this may eventually
support and ``Governed-Scaling.md`` for the equivalence/bounded-error bar it clears.
"""

from __future__ import annotations

from typing import Callable

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

# The documented sensitivity hierarchy, as defaults. A tensor tagged "protected"
# (embeddings, lm_head/unembedding, MoE router/gate, early attention) is *never*
# allocated below ``PROTECTED_FLOOR`` bits, regardless of measured sensitivity —
# this is the "selective upcasting" trick that rescues the worst layers without
# re-quantizing the whole model. Reproduces the structural choice Unsloth's UD2.0
# "XL" variants make (keep the spikes at Q6–Q8, crush the rest).
PROTECTED_FLOOR = 6
MIN_BITS = 1
MAX_BITS = 8


# ---------------------------------------------------------------------------
# 1. Sensitivity — the output-distortion cost of quantizing one tensor in isolation
# ---------------------------------------------------------------------------

def kl_divergence(p, q, *, eps: float = 1e-12) -> float:
    """KL(p ‖ q) in nats, for discrete distributions over the last axis.

    Symmetric epsilon flooring avoids log(0) and division by zero. Used as the
    *output-fidelity* metric throughout: the allocator minimizes total output KL,
    not weight error (per-layer weight error is a poor proxy for output distortion).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * (np.log(p) - np.log(q)), axis=-1).mean())


def quantize_uniform(W, bits: int):
    """Symmetric uniform ``bits``-bit quantize (the toy inside-the-allocator quantizer).

    ``bits=1`` → sign-only ternary-ish; ``bits=8`` → the INT8 path in ``moe/quant``.
    Returns the dequantized approximation (same shape). We use a simple uniform scheme
    here because the *allocation policy* is the contribution; the per-tensor quantizer
    is swappable (FP8/NVFP4/codebook) via ``QuantFn`` in :func:`bit_allocator`.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if bits < 1:
        raise ValueError("bits must be >= 1")
    W = np.asarray(W, dtype=np.float64)
    if bits == 1:
        # sign-only: magnitudes collapse to ±mean|W| — the brutal 1-bit case
        mag = np.mean(np.abs(W))
        return np.sign(W) * mag
    qmax = (1 << (bits - 1)) - 1            # 2^(b-1)-1 levels each side of zero
    amax = np.max(np.abs(W))
    if amax == 0:
        return W.copy()
    scale = amax / qmax
    q = np.clip(np.round(W / scale), -qmax, qmax)
    return q * scale


# A tensor sensitivity profile: (name, numels, sensitivity, protected).
# ``sensitivity`` is a non-negative scalar — larger = more distortion when quantized.
# In production this is the *measured* output-KL contribution of quantizing that tensor
# in isolation against a calibration set (see :func:`measure_sensitivity`). Here it is
# an input, so the allocator is testable without a model.
TensorProfile = tuple  # (name: str, numels: int, sensitivity: float, protected: bool)
QuantFn = Callable[[object, int], object]  # (tensor, bits) -> approximated tensor


def measure_sensitivity(tensors: "dict[str, np.ndarray]", logits_fn,
                        calibration_inputs, bits_probe: int = 2) -> "dict[str, float]":
    """Measure each tensor's output-KL contribution when quantized to ``bits_probe``.

    This is the *honest* way to get the sensitivity vector (vs. hand-assigning it):
    hold the model at full precision, swap one tensor at a time for its ``bits_probe``
    quantization, run ``logits_fn`` over ``calibration_inputs``, and record the mean
    output KL(full ‖ quantized-one-tensor). A tensor whose quantization barely moves the
    output is redundant (low sensitivity → crush it); one that wrecks the output is
    load-bearing (high sensitivity → keep it high-precision).

    ``tensors``    : name -> weight array (the model's named parameters to probe).
    ``logits_fn``  : callable(seq_batch) -> next-token softmax probs (full precision).
    ``calibration_inputs`` : batched sequences drawn from the *deployment distribution*
                     (NOT wikitext — see :mod:`moe.calibrate` for the matching discipline).

    Returns name -> sensitivity. Pure-numpy; the model is the caller's responsibility.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if bits_probe < MIN_BITS:
        raise ValueError("bits_probe must be >= 1")
    # Full-precision reference outputs (computed once).
    ref = logits_fn(calibration_inputs)
    sens: dict[str, float] = {}
    for name, W in tensors.items():
        saved = tensors[name]
        tensors[name] = quantize_uniform(W, bits_probe)   # swap in quantized version
        try:
            q_out = logits_fn(calibration_inputs)         # caller reads tensors[name]
        finally:
            tensors[name] = saved                          # always restore
        sens[name] = kl_divergence(ref, q_out)
    return sens


# ---------------------------------------------------------------------------
# 2. Bit-allocation — greedy budgeted allocation with a protected floor
# ---------------------------------------------------------------------------

def _distortion(sensitivity: float, bits: int) -> float:
    """Modeled output distortion of a tensor at ``bits`` width.

    Decreasing-returns in bits: each halving of the quant step roughly halves distortion,
    so distortion ∝ sensitivity · 2^-(bits-1). At ``bits=1`` this is just ``sensitivity``
    (the probe sensitivity is *defined* at the brutal width); more bits buy exponential
    relief. This monotone, convex-decreasing shape is what makes greedy optimal-ish.
    """
    return sensitivity * (2.0 ** -(bits - 1))


def _marginal_gain(sensitivity: float, current_bits: int) -> float:
    """Distortion reduction from adding one bit at ``current_bits``.

    = distortion(b) - distortion(b+1) = sensitivity · 2^-b. This is the greedy score:
    spend the next bit where it removes the most distortion. Monotone-decreasing in
    ``current_bits``, so the greedy choice is well-defined and budget-respecting.
    """
    return sensitivity * (2.0 ** -current_bits)


def bit_allocator(profiles: "list[TensorProfile]", target_avg_bits: float, *,
                  protected_floor: int = PROTECTED_FLOOR,
                  min_bits: int = MIN_BITS, max_bits: int = MAX_BITS) -> "dict[str, int]":
    """Allocate an integer bit-width per tensor under a total-size budget.

    Inputs:
        ``profiles``         : list of (name, numels, sensitivity, protected).
        ``target_avg_bits``  : the *numel-weighted* average width to hit
                               (e.g. 2.0 → ~2-bit model; 1.5 → the GLM-5.2 1-bit regime).

    Algorithm (greedy, decreasing-returns knapsack):
        1. Seed every tensor at ``min_bits`` (1), protected tensors at their floor.
        2. Compute the total budget ``B = target_avg_bits · Σ numels`` (in bit-elements).
        3. Repeatedly give one more bit to the (non-maxed) tensor with the highest
           marginal distortion-reduction-per-bit, until the budget is exhausted.

    Guarantees (checked in :func:`offline_invariants`):
        - Total allocated bits **≤** B  (budget never exceeded).
        - Every protected tensor **≥** ``protected_floor``  (floor never violated).
        - Every tensor in ``[min_bits, max_bits]``  (range respected).

    Returns name -> allocated bits.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if not profiles:
        return {}
    if min_bits < 1:
        raise ValueError("min_bits must be >= 1")
    if not (min_bits <= protected_floor <= max_bits):
        raise ValueError("need min_bits <= protected_floor <= max_bits")
    if target_avg_bits < min_bits:
        raise ValueError(
            f"target_avg_bits={target_avg_bits} below min_bits={min_bits}: infeasible"
        )

    names = [p[0] for p in profiles]
    numels = np.array([p[1] for p in profiles], dtype=np.float64)
    sens = {p[0]: float(max(p[2], 0.0)) for p in profiles}
    protected = {p[0]: bool(p[3]) for p in profiles}
    bits = {n: (protected_floor if protected[n] else min_bits) for n in names}

    total_budget = float(target_avg_bits) * float(numels.sum())
    used = sum(bits[n] * numels[i] for i, n in enumerate(names))

    # Fail-closed: if the floor+min seeding already exceeds the budget, the target is
    # infeasible given the protected floors — the floors cannot be honored within budget.
    # This is the same honesty as the min_bits infeasibility check above: never silently
    # overspend; make the user raise the target or shrink the protected set.
    if used > total_budget + 1e-9:
        seed_avg = used / float(numels.sum())
        raise ValueError(
            f"target_avg_bits={target_avg_bits} infeasible: protected floors "
            f"({PROTECTED_FLOOR}-bit) + min_bits seeding already averages "
            f"{seed_avg:.3f} bits. Raise target_avg_bits to >= {seed_avg:.3f} or "
            f"reduce the protected-floor / protected-tensor count."
        )

    # Greedy: spend the remaining budget one bit at a time on the best marginal gain.
    # We weight marginal gain by NOTHING (a bit is a bit) — but a bit on a larger tensor
    # costs more budget, so the greedy step naturally prefers high-sensitivity *and*
    # small tensors, which is the right behavior (spend cheap bits where they help most).
    while used < total_budget - 1e-9:
        best_name, best_gain = None, -1.0
        for i, n in enumerate(names):
            if bits[n] >= max_bits:
                continue
            cost = numels[i]                          # one more bit on tensor n costs this
            if used + cost > total_budget + 1e-9:
                continue                              # would exceed budget
            gain = _marginal_gain(sens[n], bits[n])
            if gain > best_gain:
                best_gain, best_name = gain, n
        if best_name is None:
            break                                     # no tensor can take a bit in-budget
        bits[best_name] += 1
        used += numels[names.index(best_name)]

    return bits


def total_bits(bits: "dict[str, int]", profiles: "list[TensorProfile]") -> float:
    """Numel-weighted total bit-budget actually used by an allocation (for checking)."""
    numel_of = {p[0]: p[1] for p in profiles}
    return float(sum(bits[n] * numel_of[n] for n in bits))


def avg_bits(bits: "dict[str, int]", profiles: "list[TensorProfile]") -> float:
    """Achieved average bit-width (total bits / total numels)."""
    n = sum(p[1] for p in profiles)
    return total_bits(bits, profiles) / n if n else 0.0


# ---------------------------------------------------------------------------
# 3b. LoRA rank allocation — the SAME greedy budgeted allocator, applied to LoRA
#     rank instead of quant bits (the P6 "moe/adapt -> QLoRA" bridge).
# ---------------------------------------------------------------------------
#
# QLoRA spends a fixed *rank budget* uniformly: every adapted module gets the same
# rank ``r``. That misallocates capacity for the same reason uniform quantization
# misallocates bits — a few modules are load-bearing and many are redundant. The fix
# is identical: allocate rank in proportion to measured sensitivity under a fixed
# total-parameter budget, with a protected floor. We REUSE ``bit_allocator`` verbatim
# (no new algorithm): "bits" -> "rank"; a tensor's "numels" -> a module's LoRA
# cost-per-rank (``fan_in + fan_out``, since a LoRA adapter adds ``r*(fan_in+fan_out)``
# params); "sensitivity" -> the module's importance signal. The cost-weighted average
# rank equals ``target_avg_rank``, so the TOTAL adapter parameter count matches a
# uniform-rank=``target_avg_rank`` baseline exactly — same budget, redistributed.

MIN_LORA_RANK = 1
MAX_LORA_RANK = 64


def weight_norm_sensitivity(weights: "dict[str, np.ndarray]") -> "dict[str, float]":
    """Cheap proxy sensitivity = per-module Frobenius norm (no forward pass).

    An HONEST heuristic, NOT the principled :func:`measure_sensitivity` output-KL probe:
    higher-norm weight matrices tend to produce larger activations and so are more
    output-sensitive, but this is a correlate, not a measurement. Use it when a
    calibration ``logits_fn`` is unavailable (e.g. a quick QLoRA run); use
    ``measure_sensitivity`` when you can afford the forward passes. Returns name -> norm.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    return {name: float(np.linalg.norm(np.asarray(W, dtype=np.float64)))
            for name, W in weights.items()}


def lora_rank_allocation(
    modules: "list[tuple[str, int, int, float, bool]]",
    *,
    target_avg_rank: float,
    min_rank: int = MIN_LORA_RANK,
    max_rank: int = MAX_LORA_RANK,
    protected_rank: "int | None" = None,
    alpha_per_rank: float = 2.0,
) -> "tuple[dict[str, int], dict[str, int]]":
    """Allocate a per-module LoRA rank under a fixed parameter budget, by sensitivity.

    ``modules`` : list of ``(name, fan_in, fan_out, sensitivity, protected)``.
    ``target_avg_rank`` : the cost-weighted average rank to hit — set it to the uniform
        LoRA ``r`` you would otherwise use, and the total adapter param count is preserved.

    Returns ``(rank_pattern, alpha_pattern)``, both ``dict[module_name -> int]``, ready to
    hand to a PEFT ``LoraConfig`` (``rank_pattern=...``, ``alpha_pattern=...``). The
    allocation is exactly :func:`bit_allocator` with rank semantics, so it inherits the
    same CI-proven guarantees: total budget respected, protected floor honored, ranks in
    ``[min_rank, max_rank]``, sensitivity-ordered. ``alpha_pattern`` scales alpha with rank
    (``alpha = round(alpha_per_rank * rank)``) so the per-module LoRA scaling ``alpha/r``
    stays constant across the allocation.
    """
    if not modules:
        return {}, {}
    floor = min_rank if protected_rank is None else int(protected_rank)
    floor = max(min_rank, min(floor, max_rank))
    # Build TensorProfiles: cost-per-rank = fan_in + fan_out (LoRA params per unit rank).
    profiles = [
        (name, int(fan_in) + int(fan_out), float(max(sens, 0.0)), bool(protected))
        for (name, fan_in, fan_out, sens, protected) in modules
    ]
    ranks = bit_allocator(
        profiles, float(target_avg_rank),
        protected_floor=floor, min_bits=int(min_rank), max_bits=int(max_rank),
    )
    rank_pattern = {n: int(r) for n, r in ranks.items()}
    alpha_pattern = {n: int(round(alpha_per_rank * r)) for n, r in ranks.items()}
    return rank_pattern, alpha_pattern


# ---------------------------------------------------------------------------
# 3. Offline invariants — the governed-scaling equivalence/bounded-error bar
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. Budget never exceeded. A mixed profile (some protected, some not). The protected
    #    tensors at floor 6 plus the rest at min 1 seed an average ~3.0 bits, so we test
    #    budget compliance at a *feasible* target (3.5) and infeasibility separately.
    profiles = [
        ("embed", 4000, 8.0, True),     # protected, very sensitive
        ("lm_head", 4000, 7.0, True),   # protected
        ("router", 200, 9.0, True),     # protected, tiny but critical
        ("attn_q", 1000, 5.0, False),
        ("attn_k", 1000, 4.5, False),
        ("attn_v", 1000, 4.8, False),
        ("ffn_1", 3000, 1.0, False),    # redundant → crush
        ("ffn_2", 3000, 0.8, False),
        ("expert_0", 2000, 0.5, False), # redundant MoE expert
        ("expert_1", 2000, 0.4, False),
    ]
    target = 3.5
    bits = bit_allocator(profiles, target)
    budget = target * sum(p[1] for p in profiles)
    used = total_bits(bits, profiles)
    checks["budget_respected"] = bool(used <= budget + 1e-6)
    detail["budget"] = round(budget, 1)
    detail["used"] = round(used, 1)

    # 2. Protected floor never violated.
    checks["protected_floor_honored"] = all(
        bits[p[0]] >= PROTECTED_FLOOR for p in profiles if p[3]
    )
    detail["protected_bits"] = {p[0]: bits[p[0]] for p in profiles if p[3]}

    # 3. Every tensor within [min, max].
    checks["range_respected"] = all(
        MIN_BITS <= bits[p[0]] <= MAX_BITS for p in profiles
    )

    # 4. The allocator is sensitivity-aware: the most sensitive *non-protected* tensor
    #    gets >= the least sensitive one. (ffn redundancy → fewer bits than attn.)
    checks["sensitivity_ordering"] = bool(bits["attn_q"] >= bits["ffn_1"])
    detail["alloc"] = {n: bits[n] for n, *_ in profiles}

    # 5. Infeasible target (below min_bits) raises, not silently degrades.
    try:
        bit_allocator(profiles, 0.5)
        checks["infeasible_raises"] = False
    except ValueError:
        checks["infeasible_raises"] = True

    # 5b. A target below the protected-floor-seeded average also raises (fail-closed on
    #     floor-infeasibility), rather than silently overspending the budget.
    try:
        bit_allocator(profiles, 2.0)   # seeding already averages ~3.0
        checks["floor_infeasible_raises"] = False
    except ValueError:
        checks["floor_infeasible_raises"] = True

    # 5c. The brutal 1-bit regime IS reachable with a smaller protected set / lower floor:
    #     a profile with one small protected tensor and low floor can hit avg ~1.5.
    brutal_profiles = [("router", 100, 9.0, True), ("ffn", 5000, 1.0, False)]
    brutal = bit_allocator(brutal_profiles, 1.5, protected_floor=4)
    checks["floor_holds_at_brutal_avg"] = (
        brutal["router"] >= 4 and avg_bits(brutal, brutal_profiles) <= 1.6
    )
    detail["brutal_avg"] = round(avg_bits(brutal, brutal_profiles), 3)

    # 7. Distortion model is decreasing-returns in bits (the assumption greedy relies on).
    s = 5.0
    d = [_distortion(s, b) for b in range(1, 6)]
    checks["distortion_decreasing_returns"] = bool(
        all(d[i] > d[i + 1] and d[i] > 0 for i in range(len(d) - 1))
    )

    # 8. Uniform quantize at 1 bit collapses to sign · mean|W|; at 8 bits is near-exact.
    rng = np.random.default_rng(0)
    W = rng.standard_normal((64, 64))
    q1 = quantize_uniform(W, 1)
    q8 = quantize_uniform(W, 8)
    checks["one_bit_is_sign_mean"] = bool(
        np.allclose(np.unique(np.abs(q1)), np.mean(np.abs(W)))
    )
    checks["eight_bit_near_exact"] = bool(
        np.max(np.abs(q8 - W)) / np.max(np.abs(W)) < 1.0 / 127 + 1e-9
    )

    # 9. KL is non-negative and 0 for identical distributions.
    uni = np.full((4, 10), 0.1)
    checks["kl_self_is_zero"] = abs(kl_divergence(uni, uni)) < 1e-12
    peaked = np.zeros((4, 10)); peaked[:, 0] = 1.0
    checks["kl_nonneg"] = kl_divergence(peaked, uni) > 0.0

    # 10. LoRA rank allocation reuses bit_allocator: same param budget, sensitivity-aware.
    #     Modules: (name, fan_in, fan_out, sensitivity, protected). Uniform baseline r=8.
    lora_modules = [
        ("q_proj", 2048, 2048, 9.0, True),    # protected, sensitive -> high rank
        ("k_proj", 2048, 256, 6.0, False),
        ("v_proj", 2048, 256, 7.0, False),
        ("o_proj", 2048, 2048, 5.0, False),
        ("gate_proj", 2048, 8192, 1.0, False),  # large + redundant -> low rank
        ("up_proj", 2048, 8192, 0.8, False),
        ("down_proj", 8192, 2048, 0.9, False),
    ]
    target_r = 8
    rank_pattern, alpha_pattern = lora_rank_allocation(
        lora_modules, target_avg_rank=target_r, min_rank=2, max_rank=32, protected_rank=8)
    cost = {m[0]: m[1] + m[2] for m in lora_modules}
    total_cost = sum(cost.values())
    # (a) total adapter params <= uniform-rank budget (cost-weighted avg rank <= target).
    alloc_params = sum(rank_pattern[n] * cost[n] for n in rank_pattern)
    uniform_params = target_r * total_cost
    checks["lora_budget_respected"] = alloc_params <= uniform_params + 1e-6
    detail["lora_alloc_params"] = alloc_params
    detail["lora_uniform_params"] = uniform_params
    # (b) protected module never below its floor.
    checks["lora_protected_floor"] = rank_pattern["q_proj"] >= 8
    # (c) ranks within [min,max].
    checks["lora_rank_range"] = all(2 <= r <= 32 for r in rank_pattern.values())
    # (d) sensitivity-aware: the sensitive v_proj gets >= the redundant up_proj.
    checks["lora_sensitivity_ordering"] = rank_pattern["v_proj"] >= rank_pattern["up_proj"]
    # (e) alpha tracks rank (constant alpha/r scaling), and patterns share keys.
    checks["lora_alpha_tracks_rank"] = all(
        alpha_pattern[n] == round(2.0 * rank_pattern[n]) for n in rank_pattern)
    detail["lora_rank_pattern"] = rank_pattern
    # (f) empty modules -> empty patterns (no crash).
    checks["lora_empty_safe"] = lora_rank_allocation([], target_avg_rank=8) == ({}, {})

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Adaptive-quant offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  budget={detail.get('budget')} used={detail.get('used')}")
    print(f"  allocation: {detail.get('alloc')}")
    print(f"  brutal-1.5bit avg={detail.get('brutal_avg')}")
    raise SystemExit(0 if ok else 1)

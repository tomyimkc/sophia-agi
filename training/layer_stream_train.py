#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""MegaTrain memory-centric *training* planner — the TRAINING mirror of ``serving/layer_stream.py``.

*Why this exists.* MegaTrain (Yuan/Sun/Sun/Ye, arXiv:2604.05091) is a **memory-centric** training
system: parameters + optimizer states live in **host memory**, the GPU is a **transient compute
engine**. Per layer it streams params *in* and gradients *out*, so only a few **stateless layer
templates** are device-resident at a time — peak device memory is ``a few layers``, NOT the whole
model. On the sophia-agi unified-memory boxes (DGX Spark GB10 / Mac M3 Ultra — Grace-class CPU + GPU
sharing one coherent LPDDR5x pool) the CPU↔GPU copy that MegaTrain fights is *milder* (one pool, no
PCIe hop), so the capacity win is the whole point: fit a several×-larger full-precision train on
owned hardware.

*What this module is.* A **pure, offline, deterministic memory + throughput PLANNER / MODEL** — the
training-side analog of ``serving/layer_stream.py``'s byte accounting and of ``tools/run_cluster_sim``
/ ``tools/cluster_schedule_sim`` (closed-form, no clock, no random, no torch, no GPU, no network). It
PROVES the byte-accounting question *before any GPU run*: **does an 8B full-precision train fit in
128 GB under double-buffered streaming?** It computes host residence (params + optimizer), peak device
working set (double-buffered layers + activations), the unified-memory parameter ceiling, and an
overlap-efficiency schedule model.

*What this module is NOT.* It does NO real training and makes **NO capability or throughput claim**
about real hardware. The overlap-efficiency number is a SCHEDULE model (overlapped wall ≈
``max(compute, transfer)``), not a measured throughput. Memory-fit ≠ trained-to-convergence; MegaTrain
solves **memory**, not **FLOPs** (training FLOPs ≈ ``6·params·tokens`` are unchanged — frontier
pretraining stays FLOP-walled). This is PLANNING ONLY. ``canClaimAGI`` stays **false**.

*Relationship to the serving mirror.* ``serving/layer_stream.py`` keeps the *current layer window*
resident to SERVE a model bigger than fast memory; this keeps the *current layer window + its grad +
optimizer slice* resident to TRAIN one. The tiering / double-buffer / byte-accounting concepts
transfer 1:1; the difference is the optimizer-state residence in host memory and the gradient-offload
half of the double buffer.

Falsifiable offline invariants (``offline_invariants()``, CI-gated, mirrors ``tests/test_layer_stream``):
  - ceiling math: 128 GB adam-fp32 ≈ 8B trainable params; 512 GB ≈ 32B (within ~10%);
  - the double-buffer peak uses ``double_buffer_depth`` layers, NOT ``n_layers`` (peak ≪ full model);
  - GaLore / LoRA shrink bytes-per-param → strictly larger ceiling than full Adam;
  - activation recomputation reduces activation bytes;
  - overlap_efficiency > 1 when overlapped (depth ≥ 2) and ≈ 1 when serial (depth 1);
  - determinism (same inputs → same outputs);
  - a 512k-context scenario shows activations dominate the device working set.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

GIB = 1024 ** 3  # bytes per GiB — the unit the box budgets are quoted in (128 GB ≈ 128 GiB here)


# ---------------------------------------------------------------------------
# Optimizer byte accounting — host residence per trainable parameter
# ---------------------------------------------------------------------------

# Each scheme's per-trainable-param host cost, in bytes. Documented inline:
#   "adam-fp32"        : the full-precision MegaTrain baseline — fp32 weight(4) + grad(4) +
#                        Adam m(4) + Adam v(4) = 16 B/param. This is the "16 bytes/param" wall.
#   "adam-bf16-master" : bf16 compute weight(2) + bf16 grad(2) + fp32 master weight(4) +
#                        fp32 m(4) + fp32 v(4) = 16 B/param — same total, different precision mix
#                        (Adam moments dominate; the bf16 weight/grad saving is offset by the fp32
#                        master copy mixed-precision needs). Kept distinct to document the trade.
#   "sgd-momentum"     : fp32 weight(4) + grad(4) + momentum buffer(4) = 12 B/param (no 2nd moment).
#   "galore"           : low-rank projected optimizer states (GaLore, arXiv:2403.03507). Weight(4) +
#                        grad(4) are full; the m/v states (8 B in Adam) are kept only in a rank-r
#                        subspace, so they shrink by ``rank_ratio`` → 4 + 4 + 8·rank_ratio B/param.
#                        ``rank_ratio`` in (0,1]; 1.0 degenerates to adam-fp32 (16 B).
#   "lora"             : only the adapter params are trainable; the frozen base is weight-only
#                        resident. We charge frozen base at 2 B/param (bf16, inference-precision) plus
#                        a trainable-adapter surcharge of 16 B/param scaled by ``lora_param_ratio``
#                        (adapter params / total params, typ. ~0.1–1%). Default ratio 0.005.


def optimizer_bytes_per_param(
    scheme: str,
    *,
    rank_ratio: float = 0.25,
    lora_param_ratio: float = 0.005,
) -> float:
    """Host bytes resident per *model* parameter for a training ``scheme``.

    Returns the AMORTIZED bytes/param (averaged over all model params) so that
    ``host_bytes(params, scheme) == params * optimizer_bytes_per_param(scheme)`` holds for every
    scheme, including LoRA where most params are frozen. See the module table for each scheme.

    ``rank_ratio``      : GaLore subspace fraction in (0, 1] — the optimizer-state shrink factor.
    ``lora_param_ratio``: fraction of params that are trainable adapters under LoRA, in (0, 1].
    """
    s = scheme.lower()
    if s == "adam-fp32":
        return 16.0  # 4 weight + 4 grad + 4 m + 4 v
    if s == "adam-bf16-master":
        return 16.0  # 2 bf16 wt + 2 bf16 grad + 4 fp32 master + 4 m + 4 v
    if s == "sgd-momentum":
        return 12.0  # 4 weight + 4 grad + 4 momentum
    if s == "galore":
        if not (0.0 < rank_ratio <= 1.0):
            raise ValueError("rank_ratio must be in (0, 1]")
        return 4.0 + 4.0 + 8.0 * rank_ratio  # weight + grad + low-rank (m,v)
    if s == "lora":
        if not (0.0 < lora_param_ratio <= 1.0):
            raise ValueError("lora_param_ratio must be in (0, 1]")
        frozen_base = 2.0                       # bf16 weight-only residence for the frozen base
        adapter = 16.0 * lora_param_ratio       # the trainable slice carries full Adam state
        return frozen_base + adapter
    raise ValueError(
        f"unknown scheme {scheme!r}; use adam-fp32 | adam-bf16-master | sgd-momentum | galore | lora"
    )


def host_bytes(
    params: int,
    scheme: str,
    *,
    rank_ratio: float = 0.25,
    lora_param_ratio: float = 0.005,
) -> int:
    """Total host (unified-memory) residence for params + optimizer state, in bytes.

    This is the MegaTrain residence: params and optimizer live in host memory for the WHOLE model
    (unlike the device side, which holds only a window). ``ceil`` so a partial byte never undercounts.
    """
    if params <= 0:
        raise ValueError("params must be positive")
    bpp = optimizer_bytes_per_param(
        scheme, rank_ratio=rank_ratio, lora_param_ratio=lora_param_ratio
    )
    return math.ceil(params * bpp)


# ---------------------------------------------------------------------------
# Activation / KV working set
# ---------------------------------------------------------------------------

def activation_bytes(
    batch: int,
    seq: int,
    hidden: int,
    n_layers: int,
    dtype_bytes: int = 2,
    *,
    recompute: bool = False,
) -> int:
    """Activation / KV working set held on device during a training step, in bytes.

    Model (deterministic, MegaTrain-style): a transformer stores ~``ACT_FACTOR`` hidden-sized
    activation tensors per layer per token for the backward pass. Without recomputation that is
    ``ACT_FACTOR · batch · seq · hidden · n_layers · dtype_bytes``. **Activation recomputation**
    (gradient checkpointing) keeps only a per-layer checkpoint and recomputes the rest in the
    backward, so the resident activation cost collapses from ``n_layers`` layers' worth to roughly
    ``sqrt(n_layers)`` checkpoints' worth — the standard O(sqrt(L)) memory trade (Chen et al. 2016).
    This term is what DOMINATES at long context (512k): it grows linearly in ``seq`` while the
    per-layer param/optimizer working set does not.
    """
    if min(batch, seq, hidden, n_layers, dtype_bytes) <= 0:
        raise ValueError("batch, seq, hidden, n_layers, dtype_bytes must be positive")
    ACT_FACTOR = 16  # hidden-sized tensors per layer per token kept for backward (attn+MLP+norm)
    per_layer = ACT_FACTOR * batch * seq * hidden * dtype_bytes
    if recompute:
        # O(sqrt(L)) checkpointing: resident checkpoints ≈ ceil(sqrt(n_layers)).
        resident_layers = math.ceil(math.sqrt(n_layers))
    else:
        resident_layers = n_layers
    return per_layer * resident_layers


# ---------------------------------------------------------------------------
# Peak device-resident working set — the KEY MegaTrain property
# ---------------------------------------------------------------------------

def peak_device_bytes(
    layer_params: int,
    scheme: str,
    activation_bytes_: int,
    double_buffer_depth: int = 2,
    *,
    rank_ratio: float = 0.25,
    lora_param_ratio: float = 0.005,
) -> int:
    """Peak GPU-resident working set under double-buffered layer streaming, in bytes.

    The MegaTrain property: device residence is ``double_buffer_depth`` layers' (params + grad) plus
    the activation working set — **NOT the whole model**. Only the streaming layer template and its
    gradient are device-resident; optimizer states stay in host memory and are applied there as
    gradients stream out. So:

        peak_device = double_buffer_depth · (layer_params·wt_bytes + layer_params·grad_bytes)
                      + activation_bytes

    ``wt_bytes`` / ``grad_bytes`` are the *device* precisions (fp32=4 for adam-fp32/sgd/galore;
    bf16=2 for adam-bf16-master / the frozen base under lora). Optimizer m/v are NOT on device — that
    is the whole point. ``double_buffer_depth`` ≥ 2 is the prefetch/compute/offload overlap (mirrors
    ``serving/layer_stream.prefetch_depth``); depth 1 is the un-overlapped serial case.
    """
    if layer_params <= 0:
        raise ValueError("layer_params must be positive")
    if double_buffer_depth < 1:
        raise ValueError("double_buffer_depth must be >= 1")
    if activation_bytes_ < 0:
        raise ValueError("activation_bytes_ must be >= 0")
    s = scheme.lower()
    # Device-resident weight + gradient precision per scheme (optimizer m/v stay host-side).
    if s in ("adam-fp32", "sgd-momentum", "galore"):
        wt_bytes, grad_bytes = 4, 4
    elif s == "adam-bf16-master":
        wt_bytes, grad_bytes = 2, 2
    elif s == "lora":
        wt_bytes, grad_bytes = 2, 2  # frozen base streams at bf16; adapter grad is tiny
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    per_layer = layer_params * (wt_bytes + grad_bytes)
    return double_buffer_depth * per_layer + activation_bytes_


# ---------------------------------------------------------------------------
# Fit check + parameter ceiling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ActivationCfg:
    """Activation working-set configuration (the long-context arm)."""

    batch: int = 1
    seq: int = 4096
    hidden: int = 4096
    dtype_bytes: int = 2
    recompute: bool = False


def fits(
    params: int,
    n_layers: int,
    budget_bytes: int,
    scheme: str,
    activation_cfg: ActivationCfg,
    double_buffer_depth: int = 2,
    *,
    rank_ratio: float = 0.25,
    lora_param_ratio: float = 0.005,
) -> dict:
    """Does a ``params``-param / ``n_layers``-layer train fit a unified ``budget_bytes``?

    Returns ``{hostBytes, peakDeviceBytes, fitsHost, fitsDevice, fits, headroomBytes}``. On a unified
    box host and device draw from the SAME pool, so the binding constraint is
    ``hostBytes + peakDeviceBytes <= budget`` — ``headroomBytes`` is the slack. ``fitsHost`` /
    ``fitsDevice`` are reported separately for diagnosis (which half is the wall).
    """
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    layer_params = max(1, params // n_layers)
    hb = host_bytes(params, scheme, rank_ratio=rank_ratio, lora_param_ratio=lora_param_ratio)
    act = activation_bytes(
        activation_cfg.batch, activation_cfg.seq, activation_cfg.hidden, n_layers,
        activation_cfg.dtype_bytes, recompute=activation_cfg.recompute,
    )
    pdb = peak_device_bytes(
        layer_params, scheme, act, double_buffer_depth,
        rank_ratio=rank_ratio, lora_param_ratio=lora_param_ratio,
    )
    # Unified pool: host residence and the device working set coexist in the same budget.
    total = hb + pdb
    return {
        "hostBytes": hb,
        "peakDeviceBytes": pdb,
        "activationBytes": act,
        "fitsHost": hb <= budget_bytes,
        "fitsDevice": pdb <= budget_bytes,
        "fits": total <= budget_bytes,
        "headroomBytes": budget_bytes - total,
    }


def ceiling_params(
    budget_bytes: int,
    scheme: str,
    *,
    rank_ratio: float = 0.25,
    lora_param_ratio: float = 0.005,
) -> int:
    """Max trainable params that fit ``budget_bytes`` of unified memory for a ``scheme``.

    The residence ceiling ``budget / bytes_per_param`` — the headline MegaTrain capacity number,
    e.g. 128 GiB adam-fp32 ≈ 8B, 512 GiB ≈ 32B. This is a MEMORY-FIT ceiling (params+optimizer
    residence), NOT a trained-to-convergence claim and NOT FLOP-aware. Ignores the (comparatively
    small, model-size-independent) device activation window — that is reported by ``fits``.
    """
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")
    bpp = optimizer_bytes_per_param(
        scheme, rank_ratio=rank_ratio, lora_param_ratio=lora_param_ratio
    )
    return int(budget_bytes // bpp)


# ---------------------------------------------------------------------------
# Double-buffer overlap — a SCHEDULE model, not a measured throughput
# ---------------------------------------------------------------------------

def overlap_efficiency(
    compute_ms: float,
    transfer_ms: float,
    double_buffer_depth: int = 2,
) -> float:
    """Schedule-model efficiency of double-buffered streaming: serial / overlapped.

    With depth ≥ 2 the prefetch/compute/offload pipeline overlaps the per-layer param transfer with
    the previous layer's compute, so the overlapped wall per layer ≈ ``max(compute, transfer)`` vs the
    serial ``compute + transfer``. Efficiency = serial / overlapped ∈ [1, 2]. Depth 1 is the
    un-overlapped case (overlapped == serial → efficiency 1.0).

    HONEST: this is a SCHEDULE model (like ``cluster_schedule_sim``'s closed form), NOT a measured
    throughput. ``compute_ms`` / ``transfer_ms`` are estimates the caller supplies; real overlap is
    gated by stream scheduling, kernel launch latency, and the 273 GB/s LPDDR5x bandwidth. No
    hardware claim.
    """
    if compute_ms < 0 or transfer_ms < 0:
        raise ValueError("compute_ms and transfer_ms must be >= 0")
    if double_buffer_depth < 1:
        raise ValueError("double_buffer_depth must be >= 1")
    serial = compute_ms + transfer_ms
    if serial == 0:
        return 1.0
    if double_buffer_depth >= 2:
        overlapped = max(compute_ms, transfer_ms)
    else:
        overlapped = serial  # no double buffer → no overlap
    if overlapped == 0:
        return 1.0
    eff = serial / overlapped
    return min(eff, 2.0)  # cap: two stages overlapped can at best halve the wall


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

# Box budgets the cluster docs reason about, in GiB (Spark 128, Mac 512, 8-Spark ≈ 1024).
BOX_BUDGETS_GIB: "tuple[tuple[str, int], ...]" = (
    ("Spark (GB10, 128GB)", 128),
    ("Mac (M3 Ultra, 512GB)", 512),
    ("8-Spark (~1TB)", 1024),
)

# A few representative model sizes (params, n_layers) for the report grid.
REPORT_MODELS: "tuple[tuple[str, int, int], ...]" = (
    ("3B", 3_000_000_000, 28),
    ("8B", 8_000_000_000, 32),
    ("32B", 32_000_000_000, 60),
    ("64B", 64_000_000_000, 80),
)


def _human_gb(b: int) -> str:
    return f"{b / GIB:7.1f}GiB"


def _human_params(p: int) -> str:
    if p >= 1_000_000_000:
        return f"{p / 1e9:.1f}B"
    return f"{p / 1e6:.1f}M"


def report(scheme: str = "adam-fp32", *, double_buffer_depth: int = 2) -> str:
    """Plain-text planning table over Spark(128) / Mac(512) / 8-Spark(1024) budgets × model sizes.

    For each box: the unified-memory parameter CEILING; for each model: fits?, host bytes, peak
    device bytes. PLANNING ONLY — these are byte-accounting estimates, no hardware claim;
    canClaimAGI=false.
    """
    act_cfg = ActivationCfg(batch=1, seq=4096, hidden=4096, dtype_bytes=2, recompute=False)
    lines = [
        f"MegaTrain offline planner — scheme={scheme}, double_buffer_depth={double_buffer_depth}, "
        f"activations: batch={act_cfg.batch} seq={act_cfg.seq} hidden={act_cfg.hidden}",
        "  (PLANNING: pure byte-accounting model; NO real training, NO hardware/capability claim. "
        "canClaimAGI=false)",
        "  (MegaTrain property: peak DEVICE bytes = double_buffer_depth layers + activations, "
        "NOT the whole model)",
        "",
    ]
    for box_name, gib in BOX_BUDGETS_GIB:
        budget = gib * GIB
        ceil_p = ceiling_params(budget, scheme)
        lines.append(f"== {box_name} | adam-fp32-equiv ceiling for scheme: "
                     f"~{_human_params(ceil_p)} trainable params ==")
        lines.append(
            f"  {'model':>6} | {'layers':>6} | {'fits?':>5} | {'host':>11} | "
            f"{'peakDevice':>11} | {'headroom':>11}"
        )
        lines.append(f"  {'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}")
        for mname, params, nlayers in REPORT_MODELS:
            r = fits(params, nlayers, budget, scheme, act_cfg, double_buffer_depth)
            lines.append(
                f"  {mname:>6} | {nlayers:>6} | {('YES' if r['fits'] else 'no'):>5} | "
                f"{_human_gb(r['hostBytes'])} | {_human_gb(r['peakDeviceBytes'])} | "
                f"{_human_gb(r['headroomBytes'])}"
            )
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    checks: "dict[str, bool]" = {}
    detail: dict = {}

    # 1. Ceiling math: 128 GiB adam-fp32 ≈ 8B; 512 GiB ≈ 32B (within ~10%).
    c128 = ceiling_params(128 * GIB, "adam-fp32")
    c512 = ceiling_params(512 * GIB, "adam-fp32")
    detail["ceiling_128gb"] = c128
    detail["ceiling_512gb"] = c512
    checks["ceiling_128gb_is_8B"] = abs(c128 - 8e9) / 8e9 <= 0.10
    checks["ceiling_512gb_is_32B"] = abs(c512 - 32e9) / 32e9 <= 0.10

    # 2. Double-buffer peak uses double_buffer_depth layers, NOT n_layers (peak << full model).
    params, nlayers = 8_000_000_000, 32
    layer_params = params // nlayers
    act = activation_bytes(1, 4096, 4096, nlayers, 2)
    pdb = peak_device_bytes(layer_params, "adam-fp32", act, double_buffer_depth=2)
    full_model_resident = host_bytes(params, "adam-fp32")
    detail["peak_device_bytes_8B"] = pdb
    detail["full_model_host_bytes_8B"] = full_model_resident
    checks["peak_uses_buffer_not_nlayers"] = pdb < full_model_resident // 4  # ≪ whole model
    # depth 2 peak's param/grad term must be exactly 2 layers' worth (not nlayers)
    two_layer = 2 * layer_params * (4 + 4)
    checks["peak_is_two_layers_plus_act"] = pdb == two_layer + act

    # 3. GaLore / LoRA shrink bytes-per-param → strictly larger ceiling than full Adam.
    base = ceiling_params(128 * GIB, "adam-fp32")
    galore = ceiling_params(128 * GIB, "galore", rank_ratio=0.25)
    lora = ceiling_params(128 * GIB, "lora", lora_param_ratio=0.005)
    detail["ceiling_galore"] = galore
    detail["ceiling_lora"] = lora
    checks["galore_shrinks_bpp"] = galore > base
    checks["lora_shrinks_bpp"] = lora > base
    checks["galore_rank1_equals_adam"] = (
        optimizer_bytes_per_param("galore", rank_ratio=1.0) == 16.0
    )

    # 4. Activation recompute reduces activation bytes.
    full_act = activation_bytes(1, 8192, 4096, 64, 2, recompute=False)
    recomp_act = activation_bytes(1, 8192, 4096, 64, 2, recompute=True)
    detail["activation_full"] = full_act
    detail["activation_recompute"] = recomp_act
    checks["recompute_reduces_activations"] = recomp_act < full_act

    # 5. overlap_efficiency > 1 when overlapped (depth ≥ 2), ≈ 1 when serial (depth 1).
    eff_overlap = overlap_efficiency(100.0, 100.0, double_buffer_depth=2)
    eff_serial = overlap_efficiency(100.0, 100.0, double_buffer_depth=1)
    detail["overlap_eff_depth2"] = eff_overlap
    detail["overlap_eff_depth1"] = eff_serial
    checks["overlap_gt_1_when_buffered"] = eff_overlap > 1.0
    checks["overlap_eq_1_when_serial"] = abs(eff_serial - 1.0) < 1e-9
    checks["overlap_capped_at_2"] = overlap_efficiency(1.0, 1.0, 2) <= 2.0

    # 6. Determinism: same inputs → same outputs.
    a = fits(8_000_000_000, 32, 128 * GIB, "adam-fp32", ActivationCfg(), 2)
    b = fits(8_000_000_000, 32, 128 * GIB, "adam-fp32", ActivationCfg(), 2)
    checks["deterministic"] = (a == b)

    # 7. 512k-context scenario: activations DOMINATE the device working set.
    long_cfg = ActivationCfg(batch=1, seq=524288, hidden=4096, dtype_bytes=2, recompute=True)
    lp = 8_000_000_000 // 32
    long_act = activation_bytes(long_cfg.batch, long_cfg.seq, long_cfg.hidden, 32,
                                long_cfg.dtype_bytes, recompute=long_cfg.recompute)
    long_param_window = 2 * lp * (4 + 4)  # depth-2 param/grad window
    detail["long_ctx_activation_bytes"] = long_act
    detail["long_ctx_param_window_bytes"] = long_param_window
    checks["long_context_activations_dominate"] = long_act > long_param_window

    # 8. THE pre-registered first deliverable: an 8B full-precision adam-fp32 train FITS in 128 GiB
    #    under double-buffered streaming. The honest result: params+optimizer residence alone is
    #    ~119 GiB (8e9·16 B), so the fit needs the composing MegaTrain lever — activation
    #    RECOMPUTATION — to keep the device activation window inside the ~9 GiB of remaining budget.
    #    Without recompute the seq=4096 activation working set tips it over 128 GiB; with recompute
    #    (and the streaming param/grad window, NOT the whole model on device) it fits with headroom.
    fit8 = fits(8_000_000_000, 32, 128 * GIB, "adam-fp32",
                ActivationCfg(recompute=True), 2)
    fit8_no_recompute = fits(8_000_000_000, 32, 128 * GIB, "adam-fp32",
                             ActivationCfg(recompute=False), 2)
    detail["fit_8B_128gb_recompute"] = fit8
    detail["fit_8B_128gb_no_recompute_headroom"] = fit8_no_recompute["headroomBytes"]
    checks["8B_fits_128gb_with_recompute"] = fit8["fits"]
    # honest: it's TIGHT — the no-recompute case does NOT fit (activations tip it over).
    checks["8B_no_recompute_is_tight"] = not fit8_no_recompute["fits"]

    # 9. Fail-closed: bad scheme, bad params, bad rank_ratio, bad depth.
    checks["fail_closed"] = True
    for bad in (
        lambda: optimizer_bytes_per_param("nope"),
        lambda: host_bytes(0, "adam-fp32"),
        lambda: optimizer_bytes_per_param("galore", rank_ratio=0.0),
        lambda: peak_device_bytes(100, "adam-fp32", 0, double_buffer_depth=0),
        lambda: ceiling_params(0, "adam-fp32"),
    ):
        try:
            bad()
            checks["fail_closed"] = False  # should have raised
            break
        except ValueError:
            pass  # expected — fail-closed held for this case

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--params", type=float, default=8e9,
                    help="trainable model params (e.g. 8e9 for 8B)")
    ap.add_argument("--layers", type=int, default=32, help="number of transformer layers")
    ap.add_argument("--budget-gb", type=float, default=128.0, help="unified memory budget in GiB")
    ap.add_argument("--scheme", default="adam-fp32",
                    help="adam-fp32 | adam-bf16-master | sgd-momentum | galore | lora")
    ap.add_argument("--double-buffer", type=int, default=2,
                    help="double-buffer depth (prefetch/compute/offload overlap; >=2 to overlap)")
    ap.add_argument("--batch", type=int, default=1, help="activation: batch size")
    ap.add_argument("--seq", type=int, default=4096, help="activation: sequence length")
    ap.add_argument("--hidden", type=int, default=4096, help="activation: hidden size")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="activation dtype bytes (bf16=2)")
    ap.add_argument("--recompute", action="store_true",
                    help="enable activation recomputation (gradient checkpointing)")
    ap.add_argument("--rank-ratio", type=float, default=0.25, help="galore subspace fraction (0,1]")
    ap.add_argument("--lora-param-ratio", type=float, default=0.005,
                    help="lora trainable-adapter fraction of total params (0,1]")
    ap.add_argument("--self-test", action="store_true", help="run offline invariants and exit")
    ap.add_argument("--report", action="store_true",
                    help="print the Spark/Mac/8-Spark x model-size planning table")
    args = ap.parse_args(argv)

    if args.self_test:
        ok, detail = offline_invariants()
        print("MegaTrain train-planner offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        print(f"  ceiling 128GiB adam-fp32: {detail['ceiling_128gb'] / 1e9:.2f}B params; "
              f"512GiB: {detail['ceiling_512gb'] / 1e9:.2f}B params")
        print(f"  8B peak DEVICE bytes {detail['peak_device_bytes_8B'] / GIB:.2f}GiB vs full-model "
              f"host {detail['full_model_host_bytes_8B'] / GIB:.1f}GiB "
              f"(device window << whole model)")
        print(f"  512k-ctx activations {detail['long_ctx_activation_bytes'] / GIB:.1f}GiB "
              f"dominate the {detail['long_ctx_param_window_bytes'] / GIB:.3f}GiB param window")
        return 0 if ok else 1

    if args.report:
        print(report(args.scheme, double_buffer_depth=args.double_buffer))
        return 0

    # Single-scenario fit check.
    try:
        params = int(args.params)
        budget = int(args.budget_gb * GIB)
        cfg = ActivationCfg(
            batch=args.batch, seq=args.seq, hidden=args.hidden,
            dtype_bytes=args.dtype_bytes, recompute=args.recompute,
        )
        r = fits(params, args.layers, budget, args.scheme, cfg, args.double_buffer,
                 rank_ratio=args.rank_ratio, lora_param_ratio=args.lora_param_ratio)
        ceil_p = ceiling_params(budget, args.scheme,
                                rank_ratio=args.rank_ratio, lora_param_ratio=args.lora_param_ratio)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    print(f"MegaTrain offline planner — {_human_params(params)} params / {args.layers} layers, "
          f"scheme={args.scheme}, budget={args.budget_gb}GiB, double_buffer={args.double_buffer}")
    print("  (PLANNING: byte-accounting model; NO real training, NO hardware claim. "
          "canClaimAGI=false)")
    print(f"  host (params+optimizer) : {_human_gb(r['hostBytes'])}")
    print(f"  peak device working set : {_human_gb(r['peakDeviceBytes'])} "
          f"(activations {_human_gb(r['activationBytes'])})")
    print(f"  fits unified budget     : {'YES' if r['fits'] else 'NO'} "
          f"(headroom {_human_gb(r['headroomBytes'])})")
    print(f"  fitsHost={r['fitsHost']}  fitsDevice={r['fitsDevice']}")
    print(f"  unified-memory ceiling  : ~{_human_params(ceil_p)} trainable params for this scheme")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

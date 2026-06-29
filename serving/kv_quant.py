# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""KV-cache quantization — INT8/INT4 KV for low-RAM long-context serving.

*Why this exists.* At 1M-token context the KV cache — not the weights — is the memory
dominator: one key+value vector per head per token. Quantizing the *cache* (INT8, or INT4
for the aggressive case) is what makes very-long context affordable to serve, and it is a
natural sibling of the weight quantization in :mod:`moe.quant` applied to a *different*
object (activations stored across decode steps, not parameters).

*What is genuinely subtle here.* A KV cache has a property weights do not: it is **shared
across requests by prefix** (see :mod:`serving.kv_cache`). So a KV quantizer must preserve
**content-addressability** — two requests with the same token prefix must hash to the same
*quantized* block, or prefix sharing silently breaks. This module quantizes per-block with
a *deterministic, content-only* scale (the block's own amax), so the quantized block is a
pure function of its tokens — prefix sharing survives quantization exactly. That is the
load-bearing invariant and it is CI-checked.

*Governance.* This clears the bounded-error bar of ``Governed-Scaling.md``: the cache
round-trip error is provably bounded (per the same ``scale/2`` argument as INT8 weights),
and prefix-sharing correctness is provably preserved (quantize is content-deterministic).
See ``docs/11-Platform/Cheap-Compute-Boundary.md`` Boundary 3.

Pure stdlib + numpy; the deployment artifact is a fused quantize-on-write / dequant-on-read
kernel in the attention path.
"""

from __future__ import annotations

import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# 1. Per-block KV quantization (content-deterministic scale)
# ---------------------------------------------------------------------------

def quantize_kv_block(KV, bits: int = 8):
    """Quantize a KV block to symmetric ``bits``-bit, content-only scale.

    ``KV``: (n_tokens, n_dim) — one cache block. The scale is ``amax(|KV|) / qmax``,
    computed *only* from the block's own values, so the quantized block is a deterministic
    function of its content. Two requests with identical tokens → identical quantized
    block → identical prefix hash. This is what keeps prefix sharing correct under
    quantization (checked in :func:`offline_invariants`).

    Returns ``(q_int, scale)`` where ``q_int`` is the integer grid and ``scale`` is the
    per-block scalar. ``bits=8`` → INT8 (the conservative serving default); ``bits=4`` →
    INT4 (8× smaller than FP16 KV, the aggressive long-context case).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if bits < 1:
        raise ValueError("bits must be >= 1")
    KV = np.asarray(KV, dtype=np.float64)
    if bits == 1:
        mag = np.mean(np.abs(KV)) if KV.size else 0.0
        return np.sign(KV).astype(np.int8), float(mag)
    qmax = (1 << (bits - 1)) - 1
    amax = float(np.max(np.abs(KV))) if KV.size else 0.0
    if amax == 0.0:
        return np.zeros_like(KV, dtype=np.int8 if bits <= 8 else np.int32), 0.0
    scale = amax / qmax
    q = np.clip(np.round(KV / scale), -qmax, qmax)
    return q.astype(np.int8 if bits <= 8 else np.int32), scale


def dequantize_kv_block(q, scale) -> "np.ndarray":
    """Reverse :func:`quantize_kv_block` — the read path in attention."""
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    return np.asarray(q, dtype=np.float64) * scale


def kv_memory_ratio(from_bits: int = 16, to_bits: int = 8) -> float:
    """Cache-size reduction: INT8 KV is 2× smaller than FP16; INT4 is 4×."""
    return from_bits / to_bits


# ---------------------------------------------------------------------------
# 2. Offline invariants — bounded error + prefix-sharing preservation
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. INT8 round-trip error bounded by scale/2 (the same guarantee as weight INT8).
    KV = rng.standard_normal((32, 16))
    q, scale = quantize_kv_block(KV, bits=8)
    dq = dequantize_kv_block(q, scale)
    checks["int8_error_bounded"] = bool(np.max(np.abs(dq - KV)) <= scale / 2 + 1e-12)
    detail["int8_max_err"] = round(float(np.max(np.abs(dq - KV))), 5)

    # 2. INT4 is 8× (4×) smaller than FP16 (FP32) and still bounded (larger step).
    q4, scale4 = quantize_kv_block(KV, bits=4)
    dq4 = dequantize_kv_block(q4, scale4)
    checks["int4_error_bounded"] = bool(np.max(np.abs(dq4 - KV)) <= scale4 / 2 + 1e-12)
    checks["int4_smaller_than_int8"] = kv_memory_ratio(16, 4) > kv_memory_ratio(16, 8)
    detail["int4_max_err"] = round(float(np.max(np.abs(dq4 - KV))), 5)
    detail["mem_ratio_int4"] = kv_memory_ratio(16, 4)

    # 3. Content-determinism: identical blocks quantize identically (prefix sharing safe).
    block_a = rng.standard_normal((16, 8))
    block_b = block_a.copy()                       # same content
    qa, sa = quantize_kv_block(block_a, bits=8)
    qb, sb = quantize_kv_block(block_b, bits=8)
    checks["identical_content_identical_quant"] = bool(
        np.array_equal(qa, qb) and sa == sb)

    # 4. Different content → (almost surely) different quantized block.
    block_c = block_a + 0.5 * rng.standard_normal(block_a.shape)
    qc, sc = quantize_kv_block(block_c, bits=8)
    checks["different_content_different_quant"] = bool(
        not np.array_equal(qa, qc) or sa != sc)

    # 5. Zero block quantizes to all-zero with scale 0 (no division blowup).
    zero = np.zeros((8, 8))
    qz, sz = quantize_kv_block(zero, bits=8)
    checks["zero_block_safe"] = bool(np.all(qz == 0) and sz == 0.0)

    # 6. bits < 1 rejected (fail-closed, no silent garbage).
    try:
        quantize_kv_block(KV, bits=0); checks["bad_bits_rejected"] = False
    except ValueError:
        checks["bad_bits_rejected"] = True

    # 7. Memory ratio math: INT8 = 2× vs FP16, INT4 = 4× vs FP16.
    checks["ratio_int8_vs_fp16"] = kv_memory_ratio(16, 8) == 2.0
    checks["ratio_int4_vs_fp16"] = kv_memory_ratio(16, 4) == 4.0

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("KV-quant offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    raise SystemExit(0 if ok else 1)

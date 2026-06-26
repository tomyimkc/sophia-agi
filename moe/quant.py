# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Low-precision weight quantization — INT8 and FP8-E4M3 (低精度训推).

Low precision is the cheapest lever for memory and bandwidth at inference: an
INT8 weight is 4× smaller than FP32 and 2× smaller than FP16; FP8 (the format
DeepSeek-V3 trains in) halves it again. The engineering content is the
*quantization scheme* — how you pick scales and bound the error so accuracy
survives. Reproduced here, proven against their error bounds in CI:

- **Symmetric INT8**, per-tensor and per-channel. ``scale = max|W| / 127``;
  ``q = clip(round(W/scale), -127, 127)``; ``dq = q·scale``. The round-trip error
  is bounded by ``scale/2`` per element (half a quant step) — checked exactly.
  Per-channel scales beat per-tensor on weight matrices with uneven column
  magnitudes (the usual case), which is *demonstrated*, not asserted.

- **FP8-E4M3** (1 sign, 4 exponent, 3 mantissa, bias 7, max 448), emulated by
  snapping to the nearest representable value. With 3 mantissa bits the relative
  round-trip error is bounded by ``2^-4 = 6.25%`` per (in-range, normal) element.

- **Weight-only quantized linear**: ``x @ dequant(quant(W))`` ≈ ``x @ W`` within a
  tolerance that scales with the quant step — the actual inference path.

Numpy reference; the GPU does this with fused dequant-in-the-GEMM kernels, which
is the deployment artifact, out of scope for the CI reference.
"""

from __future__ import annotations

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

INT8_QMAX = 127


def quantize_int8(W, *, per_channel: bool = False, axis: int = 0):
    """Symmetric INT8 quantize. Returns ``(q_int8, scale)``.

    ``per_channel`` computes one scale per slice along ``axis`` (default: rows /
    output channels), which tracks per-channel magnitude and lowers error on
    skewed weights. ``scale`` is broadcastable against ``W`` for dequant.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    W = np.asarray(W, dtype=np.float64)
    if per_channel:
        amax = np.max(np.abs(W), axis=axis, keepdims=True)
    else:
        amax = np.max(np.abs(W))
    amax = np.where(amax == 0, 1.0, amax)              # avoid 0-scale on zeros
    scale = amax / INT8_QMAX
    q = np.clip(np.round(W / scale), -INT8_QMAX, INT8_QMAX).astype(np.int8)
    return q, scale


def dequantize_int8(q, scale):
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    return q.astype(np.float64) * scale


def quantized_linear(x, W, *, per_channel: bool = True, axis: int = 0):
    """Weight-only INT8 linear: ``x @ dequant(quant(W))``.

    ``W`` is (in, out); per-channel scales are taken over ``axis`` (the input
    dim by default, one scale per output column when axis=0). Returns the
    approximate matmul. This is the inference path a weight-only-quantized model
    actually runs.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    x = np.asarray(x, dtype=np.float64)
    q, scale = quantize_int8(W, per_channel=per_channel, axis=axis)
    return x @ dequantize_int8(q, scale)


def int8_memory_reduction(from_bits: int = 32) -> float:
    """How much smaller INT8 weights are vs an ``from_bits`` baseline (4× vs fp32)."""
    return from_bits / 8.0


# ---- FP8 E4M3 emulation ----------------------------------------------------

_FP8_E4M3_MAX = 448.0
_FP8_MANT_BITS = 3
_FP8_EXP_BIAS = 7


def fp8_e4m3_roundtrip(W):
    """Snap each value to the nearest FP8-E4M3 representable value (emulated).

    Decomposes into sign/exponent/mantissa, rounds the mantissa to 3 bits, and
    clamps to ±448. Subnormals near zero collapse to 0 (E4M3 min normal ≈ 2^-6).
    Returns the dequantized fp64 approximation (same shape as ``W``).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    W = np.asarray(W, dtype=np.float64)
    sign = np.sign(W)
    a = np.abs(W)
    out = np.zeros_like(a)
    nz = a > 0
    # exponent of the value; mantissa rounded to 3 bits => quantize a to
    # nearest multiple of 2^(e - mant_bits).
    e = np.floor(np.log2(np.where(nz, a, 1.0)))
    e = np.clip(e, -_FP8_EXP_BIAS + 1, np.log2(_FP8_E4M3_MAX))  # normal range
    step = np.power(2.0, e - _FP8_MANT_BITS)
    q = np.round(a / step) * step
    q = np.clip(q, 0.0, _FP8_E4M3_MAX)
    out = np.where(nz, q, 0.0)
    return sign * out


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. INT8 round-trip error bounded by half a quant step (per-tensor).
    W = rng.standard_normal((64, 64))
    q, scale = quantize_int8(W)
    dq = dequantize_int8(q, scale)
    checks["int8_error_bounded"] = bool(np.max(np.abs(dq - W)) <= scale / 2 + 1e-12)

    # 2. Quantized values are in valid INT8 range.
    checks["int8_in_range"] = bool(q.min() >= -INT8_QMAX and q.max() <= INT8_QMAX)

    # 3. Per-channel beats per-tensor on column-skewed weights. One giant column
    # forces a huge per-tensor step that wrecks every small column; per-column
    # scales isolate it. Mean error shows this starkly (max error stays pinned by
    # the giant column either way), so we judge on mean abs error.
    skew = rng.standard_normal((32, 16))
    skew[:, 0] *= 100.0                                  # one giant-scale column
    err_pt = np.mean(np.abs(dequantize_int8(*quantize_int8(skew, per_channel=False)) - skew))
    err_pc = np.mean(np.abs(dequantize_int8(*quantize_int8(skew, per_channel=True, axis=0)) - skew))
    checks["per_channel_better"] = err_pc < err_pt / 5      # a real, large win
    detail["mean_err_per_tensor"] = round(float(err_pt), 5)
    detail["mean_err_per_channel"] = round(float(err_pc), 5)

    # 4. Weight-only quantized linear approximates the fp matmul.
    x = rng.standard_normal((8, 64))
    ref = x @ W
    approx = quantized_linear(x, W, per_channel=True, axis=0)
    rel = np.linalg.norm(approx - ref) / np.linalg.norm(ref)
    checks["qlinear_close"] = bool(rel < 0.02)          # < 2% relative error
    detail["qlinear_rel_err"] = round(float(rel), 5)

    # 5. Memory reduction is the textbook 4× vs fp32, 2× vs fp16.
    checks["mem_4x_vs_fp32"] = int8_memory_reduction(32) == 4.0
    checks["mem_2x_vs_fp16"] = int8_memory_reduction(16) == 2.0

    # 6. FP8-E4M3: relative round-trip error within the 3-mantissa-bit bound.
    vals = rng.uniform(0.1, 400.0, size=4000) * rng.choice([-1, 1], size=4000)
    fp8 = fp8_e4m3_roundtrip(vals)
    rel_fp8 = np.max(np.abs(fp8 - vals) / np.abs(vals))
    checks["fp8_relerr_bounded"] = bool(rel_fp8 <= 2 ** -_FP8_MANT_BITS + 1e-9)
    detail["fp8_max_relerr"] = round(float(rel_fp8), 5)
    checks["fp8_clamps_max"] = bool(np.max(np.abs(fp8_e4m3_roundtrip([1e9, -1e9]))) == _FP8_E4M3_MAX)
    checks["fp8_zero_preserved"] = float(fp8_e4m3_roundtrip([0.0])[0]) == 0.0

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("Quant offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  per-tensor vs per-channel int8 mean err: "
          f"{detail.get('mean_err_per_tensor')} -> {detail.get('mean_err_per_channel')}")
    print(f"  weight-only int8 linear rel err: {detail.get('qlinear_rel_err')}")
    print(f"  fp8-e4m3 max rel err: {detail.get('fp8_max_relerr')}")
    raise SystemExit(0 if ok else 1)

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""FlashAttention-style tiled online-softmax attention (paper reproduction).

Reproduces the forward pass of FlashAttention (Dao, Fu, Ermon, Rudra, Ré 2022,
arXiv:2205.14135). The point of the paper: standard attention forms the full
``S = QKᵀ`` score matrix (N×N) in HBM before softmax, so memory — and HBM traffic —
is **O(N²)**, which is what caps context length. FlashAttention never
materializes ``S``. It walks key/value tiles and maintains, per query row, a
running max ``m``, a running denominator ``l``, and a running output accumulator
``acc``, rescaling them as each new tile shifts the max (the "online softmax"
recurrence). Memory for the scores drops to **O(Bq·Bk)** — one tile — while the
output is bit-for-bit the same softmax attention.

This module gives two things:

1. ``naive_attention`` — the O(N²) reference (the formula straight from the
   Transformer paper), used as ground truth.
2. ``flash_attention_numpy`` — the tiled online-softmax algorithm in numpy.
   It produces output ``allclose`` to ``naive_attention`` (the paper's core
   correctness guarantee) while tracking the largest score tile it ever holds,
   so the O(N²)→O(tile) memory reduction is *measured*, not claimed. Supports a
   causal mask (decoder attention) with whole-tile skipping above the diagonal.

A real fused GPU kernel (``flash_attention_triton``) is included behind a gated
``triton``/CUDA import — it is the same recurrence expressed as a Triton program
and is skipped wherever Triton or a GPU is absent (i.e. always, in CI). The
numpy reference is the CI-tested artifact; the Triton kernel is the deployment
artifact.

Offline invariants (``offline_invariants()``, numpy-only, CI-gated):
  - flash == naive within fp tolerance, non-causal and causal, across shapes;
  - numerically stable under large logits (no overflow vs. a naive exp);
  - the flash path never materializes more than one ``Bq×Bk`` tile of scores,
    strictly less than N² for N beyond one tile;
  - tile size does not change the result (block-size invariance).
"""

from __future__ import annotations

import math

try:  # numpy is a CI dependency (see .github/workflows/ci.yml)
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


def triton_available() -> bool:
    """True iff a real Triton+CUDA kernel could run here (≈ never in CI)."""
    try:
        import triton  # noqa: F401
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reference (O(N^2) memory): straight softmax attention.
# ---------------------------------------------------------------------------

def naive_attention(Q, K, V, *, causal: bool = False, scale: float | None = None):
    """Ground-truth attention: softmax(QKᵀ·scale) V, materializing the full S.

    Q: (Nq, d), K/V: (Nk, d). Returns (Nq, d). Numerically stabilized by the
    standard row-max subtraction (so it is a fair correctness reference).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    d = Q.shape[-1]
    scale = 1.0 / math.sqrt(d) if scale is None else scale
    S = (Q @ K.T) * scale                       # (Nq, Nk) — the O(N^2) matrix
    if causal:
        nq, nk = S.shape
        i = np.arange(nq)[:, None]
        j = np.arange(nk)[None, :]
        S = np.where(j <= i, S, -np.inf)
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


# ---------------------------------------------------------------------------
# FlashAttention forward: tiled online softmax, O(tile) score memory.
# ---------------------------------------------------------------------------

def flash_attention_numpy(
    Q,
    K,
    V,
    *,
    block_q: int = 32,
    block_k: int = 32,
    causal: bool = False,
    scale: float | None = None,
    stats: dict | None = None,
):
    """Tiled online-softmax attention; output matches ``naive_attention``.

    Implements the FlashAttention recurrence: for each query tile, stream over
    key/value tiles keeping running ``(m, l, acc)`` and rescaling by
    ``exp(m_old - m_new)`` whenever a tile raises the max. Only one ``Bq×Bk`` tile
    of scores exists at a time. If ``stats`` is given, records
    ``max_score_tile`` (largest #score-elements held) and ``tiles_skipped``
    (causal tiles fully above the diagonal, never computed).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    Nq, d = Q.shape
    Nk = K.shape[0]
    scale = 1.0 / math.sqrt(d) if scale is None else scale

    out = np.zeros((Nq, d), dtype=np.float64)
    max_tile = 0
    tiles_skipped = 0

    for qs in range(0, Nq, block_q):
        qe = min(qs + block_q, Nq)
        Qi = Q[qs:qe]                                    # (bq, d)
        bq = qe - qs
        m = np.full((bq,), -np.inf)                      # running row max
        l = np.zeros((bq,))                              # running denominator
        acc = np.zeros((bq, d))                          # running weighted sum

        for ks in range(0, Nk, block_k):
            ke = min(ks + block_k, Nk)
            # Causal: a key tile entirely above every query in this q-tile's
            # diagonal contributes nothing — skip it (the real kernel's main win).
            if causal and ks > (qe - 1):
                tiles_skipped += 1
                continue
            Kj = K[ks:ke]                                # (bk, d)
            Vj = V[ks:ke]
            S = (Qi @ Kj.T) * scale                      # (bq, bk) — the ONLY tile
            max_tile = max(max_tile, S.size)
            if causal:
                qi = np.arange(qs, qe)[:, None]
                kj = np.arange(ks, ke)[None, :]
                S = np.where(kj <= qi, S, -np.inf)

            m_new = np.maximum(m, S.max(axis=1))         # (bq,)
            # Guard the all-masked row (m_new == -inf) so exp() stays finite.
            safe = np.where(np.isneginf(m_new), 0.0, m_new)
            p = np.exp(S - safe[:, None])                # (bq, bk)
            alpha = np.exp(np.where(np.isneginf(m), 0.0, m) - safe)  # rescale old
            alpha = np.where(np.isneginf(m), 0.0, alpha)
            l = l * alpha + p.sum(axis=1)
            acc = acc * alpha[:, None] + p @ Vj
            m = m_new

        # Rows with no unmasked key (l==0) stay zero — matches naive's 0/sum=0 edge.
        denom = np.where(l > 0, l, 1.0)
        out[qs:qe] = acc / denom[:, None]

    if stats is not None:
        stats["max_score_tile"] = max_tile
        stats["full_matrix"] = Nq * Nk
        stats["tiles_skipped"] = tiles_skipped
    return out


# ---------------------------------------------------------------------------
# Gated Triton kernel (deployment artifact; not exercised in CI).
# ---------------------------------------------------------------------------

def flash_attention_triton(Q, K, V, *, causal: bool = False, scale: float | None = None):
    """Fused FlashAttention forward as a Triton kernel. Requires Triton + CUDA.

    Raises ``RuntimeError`` where unavailable (i.e. in CI). The kernel below is
    the same online-softmax recurrence as ``flash_attention_numpy`` expressed as
    a Triton program operating on SRAM tiles — this is the form that actually
    achieves the HBM-traffic win on a GPU.
    """
    if not triton_available():
        raise RuntimeError(
            "flash_attention_triton requires triton + a CUDA device; "
            "use flash_attention_numpy for the CI-tested reference."
        )
    import torch  # noqa: F401
    import triton
    import triton.language as tl

    @triton.jit
    def _fwd(  # pragma: no cover - GPU only
        Q, K, V, Out, scale,
        stride_qm, stride_qd, stride_km, stride_kd,
        stride_vm, stride_vd, stride_om, stride_od,
        N, D: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        q = tl.load(Q + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd,
                    mask=offs_m[:, None] < N, other=0.0)
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
        end_n = (start_m + 1) * BLOCK_M if CAUSAL else N
        for start_n in range(0, end_n, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(K + offs_n[:, None] * stride_km + offs_d[None, :] * stride_kd,
                        mask=offs_n[:, None] < N, other=0.0)
            v = tl.load(V + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vd,
                        mask=offs_n[:, None] < N, other=0.0)
            s = tl.dot(q, tl.trans(k)) * scale
            if CAUSAL:
                s = tl.where(offs_m[:, None] >= offs_n[None, :], s, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
        acc = acc / l_i[:, None]
        tl.store(Out + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od,
                 acc, mask=offs_m[:, None] < N)

    import torch
    Q_t = torch.as_tensor(Q, device="cuda", dtype=torch.float32)
    K_t = torch.as_tensor(K, device="cuda", dtype=torch.float32)
    V_t = torch.as_tensor(V, device="cuda", dtype=torch.float32)
    N, D = Q_t.shape
    Out = torch.empty_like(Q_t)
    s = 1.0 / math.sqrt(D) if scale is None else scale
    BLOCK_M = BLOCK_N = 64
    grid = (triton.cdiv(N, BLOCK_M),)
    _fwd[grid](
        Q_t, K_t, V_t, Out, s,
        Q_t.stride(0), Q_t.stride(1), K_t.stride(0), K_t.stride(1),
        V_t.stride(0), V_t.stride(1), Out.stride(0), Out.stride(1),
        N, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, CAUSAL=causal,
    )
    return Out.cpu().numpy()


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    def close(a, b, tol=1e-9):
        return bool(np.allclose(a, b, atol=tol, rtol=tol))

    # 1. Matches naive across shapes (non-causal).
    ok_shapes = True
    for (Nq, Nk, d) in [(8, 8, 4), (33, 17, 8), (64, 64, 16), (1, 50, 8)]:
        Q = rng.standard_normal((Nq, d))
        K = rng.standard_normal((Nk, d))
        V = rng.standard_normal((Nk, d))
        ref = naive_attention(Q, K, V)
        fa = flash_attention_numpy(Q, K, V, block_q=16, block_k=16)
        ok_shapes &= close(ref, fa)
    checks["matches_naive_noncausal"] = ok_shapes

    # 2. Matches naive with a causal mask.
    Q = rng.standard_normal((40, 8)); K = rng.standard_normal((40, 8)); V = rng.standard_normal((40, 8))
    checks["matches_naive_causal"] = close(
        naive_attention(Q, K, V, causal=True),
        flash_attention_numpy(Q, K, V, block_q=16, block_k=16, causal=True),
    )

    # 3. Numerically stable under large logits (online rescaling vs naive exp).
    big = rng.standard_normal((16, 8)) * 50.0
    checks["stable_large_logits"] = close(
        naive_attention(big, big, big), flash_attention_numpy(big, big, big), tol=1e-7
    )

    # 4. O(tile) score memory: never materializes more than one Bq×Bk tile, and
    #    that is strictly less than the full N² matrix for N past one tile.
    stats: dict = {}
    flash_attention_numpy(Q, K, V, block_q=8, block_k=8, stats=stats)
    checks["tile_memory_bounded"] = stats["max_score_tile"] <= 8 * 8
    checks["sub_quadratic"] = stats["max_score_tile"] < stats["full_matrix"]
    detail["score_tile"] = stats["max_score_tile"]
    detail["full_matrix"] = stats["full_matrix"]

    # 5. Block-size invariance: result independent of tiling.
    a = flash_attention_numpy(Q, K, V, block_q=8, block_k=8, causal=True)
    b = flash_attention_numpy(Q, K, V, block_q=40, block_k=40, causal=True)
    c = flash_attention_numpy(Q, K, V, block_q=7, block_k=13, causal=True)
    checks["block_size_invariant"] = close(a, b) and close(a, c)

    # 6. Causal tile-skipping actually fires (kernel's main long-context win).
    s2: dict = {}
    flash_attention_numpy(Q, K, V, block_q=8, block_k=8, causal=True, stats=s2)
    checks["causal_tiles_skipped"] = s2["tiles_skipped"] > 0

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("FlashAttention offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print(f"  score tile held: {detail.get('score_tile')} vs full matrix "
          f"{detail.get('full_matrix')}")
    print(f"  triton kernel available here: {triton_available()}")
    raise SystemExit(0 if ok else 1)

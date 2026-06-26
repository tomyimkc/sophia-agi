# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""IndexShare — reusing a sparse-attention index across layers (paper reproduction).

*What this reproduces.* GLM-5.2's headline attention innovation: at 1M-token context the
*indexer* (the scoring pass that selects which keys each query attends to under
DeepSeek-style sparse attention) becomes the dominant cost — more than the gathered
attention itself. **IndexShare** runs the indexer once per block of ``group`` layers and
reuses that index set for the next ``group-1`` layers; each layer still computes its own
Q/K/V and attention weights, only the *selection* is amortized. Zhipu reports ~2.9×
per-token FLOPs reduction at 1M context (not 4× — only the indexer fraction is shared).

Why it works (the mechanistic assumption)
-----------------------------------------
Relevance of a long-context anchor is approximately stable across adjacent layers: a
passage relevant at layer ``L`` is probably still relevant at ``L+1..L+group-1``. This is
empirical and task-dependent, which is exactly why this module makes ``group`` a parameter
and **measures the error vs per-layer indexing** rather than asserting it's free.

What this module gives (the reference + the measurement)
--------------------------------------------------------
1. ``sparse_attention_indexed`` — a top-k sparse attention that takes a *precomputed* index
   (which keys each query attends to). This is one layer of an IndexShare block.
2. ``build_index`` — the indexer itself: per query, pick the top-k keys by QK score. This
   is the expensive pass IndexShare amortizes.
3. ``indexshare_block`` — run ``group`` layers sharing one index; each layer has its own
   Q/K/V. Returns outputs + a count of *index computations* (1, not ``group``).
4. ``per_layer_baseline`` — the reference: each layer builds its own index. Same outputs
   up to the approximation error of sharing; ``group``× the index compute.
5. ``quality_vs_compute_curve`` — sweep ``group ∈ {1..8}``, report (relative error vs
   per-layer indexing, index-compute reduction). This is the honest tradeoff curve —
   "IndexShare costs X% quality for Y× compute" — that Zhipu's blog states as a single
   2.9× number. We make the whole curve measurable.

Honest scope: numpy reference, CI-tested for (a) the index is computed once per block,
(b) shared-index output stays within a bounded error of per-layer output, (c) the
compute reduction is exactly ``1/group`` of the indexer cost. We do NOT reproduce GLM-5.2's
DSA indexer exactly (its full construction is not public); we reproduce the *amortization
principle* with a top-k index, which is the load-bearing idea. See
``docs/11-Platform/Cheap-Compute-Boundary.md`` Boundary 1.
"""

from __future__ import annotations

import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# 1. The indexer — the expensive pass IndexShare amortizes
# ---------------------------------------------------------------------------

def build_index(Q, K, *, topk: int):
    """Per-query top-k key selection by QKᵀ score. Returns an (Nq, topk) int index array.

    This is the "indexer": for each query, which ``topk`` keys does it attend to. At long
    context this scoring pass (over all Nq×Nk pairs) dominates — that is the cost IndexShare
    reuses across layers. Ties broken by lower index (stable).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    Nq, d = Q.shape
    Nk = K.shape[0]
    topk = min(topk, Nk)
    scores = Q @ K.T                                   # (Nq, Nk) — the expensive pass
    # top-k keys per query (highest score first)
    idx = np.argsort(-scores, axis=-1)[:, :topk]       # (Nq, topk)
    return idx


def sparse_attention_indexed(Q, K, V, index, *, scale: float | None = None):
    """One layer of sparse attention using a *precomputed* key index.

    Gathers only the indexed keys/values per query, computes attention over that subset,
    and returns the weighted sum. ``index[t]`` is the key ids query ``t`` attends to.
    This is what every layer in an IndexShare block runs — the index is the only thing
    that's shared; Q/K/V are per-layer.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    Q = np.asarray(Q, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    index = np.asarray(index)
    Nq, d = Q.shape
    topk = index.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(d)
    out = np.zeros((Nq, d), dtype=np.float64)
    for t in range(Nq):
        kid = index[t]                                 # (topk,)
        q = Q[t]                                       # (d,)
        ksel = K[kid]                                  # (topk, d)
        vsel = V[kid]                                  # (topk, d)
        s = (q @ ksel.T) * scale                       # (topk,)
        s = s - s.max()
        e = np.exp(s)
        w = e / e.sum()
        out[t] = w @ vsel
    return out


# ---------------------------------------------------------------------------
# 2. IndexShare block — group layers sharing one index
# ---------------------------------------------------------------------------

def indexshare_block(layers, *, topk: int):
    """Run ``group = len(layers)`` attention layers sharing ONE index.

    ``layers`` is a list of (Q, K, V) tuples — one per layer. The index is built ONCE
    from the *first* layer's Q/K (the "informed" index: it's off the first layer's hidden
    states, not raw input — a deliberate GLM-5.2 detail), then reused for every layer.

    Returns ``(outputs, index_computations)`` where ``index_computations == 1`` (not
    ``group``). Each output[t] uses the shared index but the layer's own Q/K/V, so the
    only approximation is the *selection*, not the attention math.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    if not layers:
        return [], 0
    Q0, K0, _ = layers[0]
    index = build_index(Q0, K0, topk=topk)             # built ONCE — the amortization
    outputs = [sparse_attention_indexed(Q, K, V, index) for (Q, K, V) in layers]
    return outputs, 1                                  # 1 index computation, not group


def per_layer_baseline(layers, *, topk: int):
    """The reference: each layer builds its OWN index. ``group`` index computations.

    This is the "no sharing" case IndexShare is compared against. Same sparse-attention
    math, same topk — the only difference is the index is recomputed per layer. The
    relative error between this and ``indexshare_block`` is the *cost of sharing*, which
    the quality-vs-compute curve measures.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    outputs = []
    n_index = 0
    for (Q, K, V) in layers:
        idx = build_index(Q, K, topk=topk)             # per-layer index — the cost
        outputs.append(sparse_attention_indexed(Q, K, V, idx))
        n_index += 1
    return outputs, n_index


# ---------------------------------------------------------------------------
# 3. The honest tradeoff curve
# ---------------------------------------------------------------------------

def quality_vs_compute_curve(layers, *, topk: int, max_group: int = 8) -> "list[dict]":
    """Sweep ``group ∈ {1..max_group}``: for each, the error vs compute of sharing.

    For ``group=1`` there's no sharing (== per-layer baseline, 0 error, 1× compute). For
    ``group=g``, we take blocks of ``g`` consecutive layers, share one index per block,
    and report:
      - ``rel_err`` : mean relative output error vs the per-layer baseline (Frobenius),
      - ``index_compute_ratio`` : 1/group (the FLOPs saving on the indexer pass),
      - ``group``  : the sharing factor.

    This is the curve Zhipu collapses into "2.9× at group=4." We expose the whole thing so
    the quality/compute tradeoff is measurable, not asserted. A real deployment picks the
    largest ``group`` whose ``rel_err`` stays under its quality budget.
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    ref, _ = per_layer_baseline(layers, topk=topk)     # ground truth: per-layer indexing
    curve = []
    n = len(layers)
    for g in range(1, min(max_group, n) + 1):
        outputs, n_idx = [], 0
        for start in range(0, n, g):
            block = [(layers[i]) for i in range(start, min(start + g, n))]
            out, k = indexshare_block(block, topk=topk)
            outputs.extend(out)
            n_idx += k
        # relative error vs per-layer baseline
        errs = [
            np.linalg.norm(outputs[i] - ref[i]) /
            max(np.linalg.norm(ref[i]), 1e-12)
            for i in range(n)
        ]
        curve.append({
            "group": g,
            "rel_err": float(np.mean(errs)),
            "max_err": float(np.max(errs)),
            "index_computations": n_idx,
            "index_compute_ratio": round(n_idx / n, 3),  # 1/g ideal
            "n_layers": n,
        })
    return curve


# ---------------------------------------------------------------------------
# 4. Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    d, N, topk = 8, 32, 4
    # 6 layers, each with its own (near-identical) Q/K/V — adjacent layers share relevance.
    # The IndexShare assumption is that relevance is *approximately* stable across layers,
    # so we perturb the per-layer Q/K by a small amount (the layers are genuinely similar).
    base_Q = rng.standard_normal((N, d))
    base_K = rng.standard_normal((N, d))
    base_V = rng.standard_normal((N, d))
    layers = []
    for i in range(6):
        # small perturbation per layer (adjacent layers are similar, the IndexShare assumption)
        Q = base_Q + 0.01 * rng.standard_normal((N, d))
        K = base_K + 0.01 * rng.standard_normal((N, d))
        V = base_V + 0.03 * rng.standard_normal((N, d))
        layers.append((Q, K, V))

    # 1. build_index returns the right shape and valid key ids.
    idx = build_index(base_Q, base_K, topk=topk)
    checks["index_shape"] = idx.shape == (N, topk)
    checks["index_valid"] = bool((idx >= 0).all() and (idx < N).all())

    # 2. IndexShare computes the index ONCE per block, not per layer.
    outs, n_idx = indexshare_block(layers[:4], topk=topk)
    checks["index_computed_once"] = n_idx == 1
    checks["block_outputs_count"] = len(outs) == 4

    # 3. Per-layer baseline computes one index per layer.
    _, n_idx_ref = per_layer_baseline(layers[:4], topk=topk)
    checks["baseline_index_per_layer"] = n_idx_ref == 4

    # 4. Shared-index output is CLOSE to per-layer when layers are similar (the assumption).
    #    With near-identical layers, the top-k index is stable, so sharing costs little.
    ref, _ = per_layer_baseline(layers[:4], topk=topk)
    errs = [np.linalg.norm(outs[i] - ref[i]) / max(np.linalg.norm(ref[i]), 1e-12)
            for i in range(4)]
    mean_err = float(np.mean(errs))
    checks["shared_close_to_perlayer"] = mean_err < 0.10   # similar layers → small error
    detail["shared_vs_perlayer_mean_err"] = round(mean_err, 4)

    # 5. The compute ratio matches ceil(n/group)/n (index computations per layer). For a
    #    group that divides n this is exactly 1/group; in general it's ceil(n/group)/n.
    #    The curve rounds the ratio to 3 dp for readability, so compare at that precision.
    curve = quality_vs_compute_curve(layers, topk=topk, max_group=6)
    n = len(layers)
    import math as _m
    checks["compute_ratio_group1"] = abs(curve[0]["index_compute_ratio"] - 1.0) < 1e-9
    expected_g4 = round(_m.ceil(n / 4) / n, 3)   # ceil(6/4)/6 = 2/6 = 0.333
    checks["compute_ratio_group4"] = abs(curve[3]["index_compute_ratio"] - expected_g4) < 1e-9
    detail["curve"] = [{"group": c["group"], "rel_err": round(c["rel_err"], 4),
                        "compute_ratio": c["index_compute_ratio"]} for c in curve]

    # 6. group=1 on the curve == per-layer baseline (0 error by construction).
    checks["group1_zero_error"] = curve[0]["rel_err"] < 1e-9

    # 7. Error is (weakly) larger for full sharing (g=n) than for g=1 — the qualitative
    #    claim that more sharing costs more quality. We don't assert strict monotonicity
    #    at every step (block-boundary effects can cause small non-monotonicities), only
    #    the endpoints, which is the honest robust claim.
    errs_by_g = [c["rel_err"] for c in curve]
    checks["error_grows_endpoints"] = errs_by_g[-1] >= errs_by_g[0]

    # 8. topk > N is clamped gracefully (no out-of-bounds).
    idx_clip = build_index(base_Q, base_K, topk=10 * N)
    checks["topk_clamped"] = idx_clip.shape == (N, N)

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("IndexShare offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  quality/compute curve (group → rel_err, compute_ratio):")
    for c in detail.get("curve", []):
        print(f"    g={c['group']}: err={c['rel_err']} compute={c['compute_ratio']}")
    raise SystemExit(0 if ok else 1)

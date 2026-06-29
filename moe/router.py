# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Top-k Mixture-of-Experts routing with capacity + load-balancing loss.

MoE is how a model gets enormous parameter count at a small *active* FLOP budget:
each token is routed to only ``k`` of ``E`` experts. The engineering pain is
**load balance** — if the router collapses onto a few experts, the rest are dead
weight and the dispatch is lopsided across devices. Two mechanisms fix it, and
both are reproduced here exactly:

1. **Capacity + dispatch.** Each expert has a fixed buffer
   ``C = ceil(capacity_factor · T · k / E)``. Tokens beyond an expert's capacity
   are *dropped* (they fall back to the residual), which keeps every expert's
   compute bounded and uniform — essential for the static shapes a real
   all-to-all dispatch needs. Dropped-token count is reported.

2. **Load-balancing auxiliary loss** (Switch Transformer, Fedus et al. 2021,
   eq. 4): ``aux = E · Σ_e f_e · P_e`` where ``f_e`` is the fraction of tokens
   *dispatched* to expert ``e`` and ``P_e`` is the mean router *probability* of
   ``e``. It is minimized (== 1.0 for top-1) under uniform routing and grows when
   routing skews, providing the gradient that spreads load.

Numpy reference, proven against these definitions in CI (``offline_invariants``).
A real system fuses the grouped expert GEMM + all-to-all on GPU; that is the
deployment artifact, out of scope for the CI reference.
"""

from __future__ import annotations

import math

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def top_k_gating(logits, k: int):
    """Softmax router → top-k experts per token with renormalized combine weights.

    Returns ``(expert_idx, combine_w, probs)``:
      - ``expert_idx``  (T, k) int : selected expert ids, highest gate first;
      - ``combine_w``   (T, k)     : gate weights of the chosen experts,
                                     renormalized to sum to 1 per token;
      - ``probs``       (T, E)     : full softmax router distribution (for the
                                     auxiliary loss' ``P_e`` term).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    logits = np.asarray(logits, dtype=np.float64)
    T, E = logits.shape
    if not (1 <= k <= E):
        raise ValueError("need 1 <= k <= E")
    probs = _softmax(logits, axis=-1)
    # top-k by probability (argsort desc, take first k)
    idx = np.argsort(-probs, axis=-1)[:, :k]                  # (T, k)
    gathered = np.take_along_axis(probs, idx, axis=-1)        # (T, k)
    combine = gathered / gathered.sum(axis=-1, keepdims=True)  # renormalize
    return idx, combine, probs


def load_balancing_loss(expert_idx, probs, num_experts: int):
    """Switch-Transformer aux loss ``E · Σ_e f_e·P_e`` (top-1 uses column 0).

    ``f_e`` = fraction of (token, slot) assignments routed to ``e``;
    ``P_e`` = mean router probability of ``e`` over tokens. Uniform routing →
    f_e = P_e = 1/E → aux = E · E · (1/E)(1/E) = 1.0 (the per-expert floor).
    """
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required")
    expert_idx = np.asarray(expert_idx)
    probs = np.asarray(probs, dtype=np.float64)
    assignments = expert_idx.reshape(-1)
    f = np.bincount(assignments, minlength=num_experts).astype(np.float64)
    f /= f.sum()                                  # dispatch fraction per expert
    P = probs.mean(axis=0)                         # mean router prob per expert
    return float(num_experts * np.sum(f * P))


class MoERouter:
    """A complete top-k MoE layer: route → capacity-limited dispatch → combine."""

    def __init__(
        self,
        num_experts: int,
        *,
        k: int = 2,
        capacity_factor: float = 1.25,
        seed: int = 0,
    ) -> None:
        if num_experts < 2:
            raise ValueError("need >= 2 experts")
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self._rng = np.random.default_rng(seed) if _HAVE_NUMPY else None

    def capacity(self, n_tokens: int) -> int:
        return math.ceil(self.capacity_factor * n_tokens * self.k / self.num_experts)

    def route(self, logits):
        """Route tokens; return a plan with per-expert token lists + drop stats.

        Honors expert capacity in priority order (highest combine weight first),
        so a token's strongest expert is filled before its weaker ones — and an
        over-capacity assignment is dropped, not silently overflowed.
        """
        idx, combine, probs = top_k_gating(logits, self.k)
        T = logits.shape[0]
        C = self.capacity(T)
        counts = [0] * self.num_experts
        # expert -> list of (token, weight); position -> kept mask
        dispatch: dict[int, list[tuple[int, float]]] = {e: [] for e in range(self.num_experts)}
        kept = np.zeros((T, self.k), dtype=bool)
        dropped = 0
        for t in range(T):
            for slot in range(self.k):                 # highest weight first
                e = int(idx[t, slot])
                if counts[e] < C:
                    counts[e] += 1
                    dispatch[e].append((t, float(combine[t, slot])))
                    kept[t, slot] = True
                else:
                    dropped += 1
        aux = load_balancing_loss(idx, probs, self.num_experts)
        return {
            "expert_idx": idx,
            "combine_w": combine,
            "kept": kept,
            "dispatch": dispatch,
            "counts": counts,
            "capacity": C,
            "dropped": dropped,
            "aux_loss": aux,
            "probs": probs,
        }

    def forward(self, x, expert_fns):
        """Run the MoE layer: dispatch x to experts, combine weighted outputs.

        ``expert_fns[e](tokens)`` maps an (m, d) array to (m, d). Tokens whose
        assignment was dropped contribute nothing from that slot (residual-only),
        matching real capacity-drop semantics.
        """
        if not _HAVE_NUMPY:
            raise RuntimeError("numpy required")
        x = np.asarray(x, dtype=np.float64)
        logits = self._router_logits(x)
        plan = self.route(logits)
        out = np.zeros_like(x)
        for e, items in plan["dispatch"].items():
            if not items:
                continue
            toks = np.array([t for t, _ in items])
            w = np.array([wt for _, wt in items])[:, None]
            y = np.asarray(expert_fns[e](x[toks]), dtype=np.float64)
            np.add.at(out, toks, w * y)
        plan["output"] = out
        return out, plan

    def _router_logits(self, x):
        d = x.shape[1]
        if not hasattr(self, "_Wg") or self._Wg.shape != (d, self.num_experts):
            self._Wg = self._rng.standard_normal((d, self.num_experts)) / math.sqrt(d)
        return x @ self._Wg


# ---------------------------------------------------------------------------
# Offline invariants
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    rng = np.random.default_rng(0)
    checks: dict[str, bool] = {}
    detail: dict = {}

    # 1. Gating: probs sum to 1, exactly k experts, combine weights sum to 1.
    logits = rng.standard_normal((20, 8))
    idx, combine, probs = top_k_gating(logits, k=2)
    checks["probs_normalized"] = bool(np.allclose(probs.sum(1), 1.0))
    checks["selects_k"] = idx.shape == (20, 2) and (np.unique(idx[0]).size == 2)
    checks["combine_normalized"] = bool(np.allclose(combine.sum(1), 1.0))

    # 2. Aux loss floor: uniform routing → aux == 1.0; skew → aux > 1.0.
    uniform = np.zeros((64, 4))                     # equal logits → uniform probs
    idx_u, _, p_u = top_k_gating(uniform, k=1)
    # spread the (tied) assignments uniformly to simulate balanced dispatch
    idx_u = np.arange(64).reshape(-1, 1) % 4
    checks["aux_floor_uniform"] = abs(load_balancing_loss(idx_u, p_u, 4) - 1.0) < 1e-9
    skew_logits = np.zeros((64, 4)); skew_logits[:, 0] = 10.0   # everyone → expert 0
    idx_s, _, p_s = top_k_gating(skew_logits, k=1)
    aux_skew = load_balancing_loss(idx_s, p_s, 4)
    checks["aux_penalizes_skew"] = aux_skew > 1.0 + 1e-6
    detail["aux_skew"] = round(aux_skew, 3)

    # 3. Capacity: no expert exceeds C; with a generous factor, zero drops.
    r = MoERouter(4, k=2, capacity_factor=2.0, seed=1)
    plan = r.route(rng.standard_normal((40, 4)) @ rng.standard_normal((4, 4)))
    checks["capacity_respected"] = all(c <= plan["capacity"] for c in plan["counts"])
    rtight = MoERouter(4, k=2, capacity_factor=0.5, seed=1)
    plan2 = rtight.route(np.tile([10.0, 0, 0, 0], (40, 1)))  # all want expert 0
    checks["overflow_dropped"] = plan2["dropped"] > 0
    checks["tight_capacity_respected"] = all(
        c <= plan2["capacity"] for c in plan2["counts"]
    )

    # 4. Identity experts reconstruct kept tokens (dispatch/combine correctness).
    r2 = MoERouter(4, k=2, capacity_factor=4.0, seed=2)   # huge capacity → no drops
    x = rng.standard_normal((30, 6))
    out, plan3 = r2.forward(x, [lambda t: t] * 4)         # every expert = identity
    # identity experts + combine weights summing to 1 ⇒ output == input
    checks["identity_reconstructs"] = bool(np.allclose(out, x, atol=1e-9))
    checks["no_drops_high_capacity"] = plan3["dropped"] == 0

    # 5. Determinism.
    a = MoERouter(4, seed=5).route(np.ones((10, 4)))
    b = MoERouter(4, seed=5).route(np.ones((10, 4)))
    checks["deterministic"] = a["counts"] == b["counts"] and a["dropped"] == b["dropped"]

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


if __name__ == "__main__":
    ok, detail = offline_invariants()
    print("MoE router offline invariants:", "PASS" if ok else "FAIL")
    for k, v in detail["checks"].items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("  aux loss under full skew:", detail.get("aux_skew"))
    raise SystemExit(0 if ok else 1)

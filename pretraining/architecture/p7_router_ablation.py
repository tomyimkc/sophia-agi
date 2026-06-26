#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""P7 — Router-policy ablation on FIXED nano-MoE experts (known-entropy floor).

The P7 "bridge" the roadmap asks for is to connect ``moe/router.py::MoERouter``
(the production-grade numpy routing layer: capacity-drop + Switch-Transformer
aux loss) to a trainable nano MoE. The honest way to do that *without inventing
a new training procedure* is an **ablation**: train a nano MoE (``MoELM``) once,
freeze its experts, then hold the **gating logits fixed** and vary only the
*routing policy* that consumes them. Routing policy becomes the single
controlled variable, measured against a closed-form entropy floor.

Three policies are compared on the *same* frozen experts and the *same* gating
logits:

1. ``handrolled-top1`` — the policy ``MoELM`` itself uses: argmax over router
   logits → exactly one expert per token, no capacity, no drops. This is a
   faithful replica of ``MoELM.forward``; its held-out loss MUST equal
   ``MoELM``'s (a self-consistency invariant).
2. ``moerouter-top1-nodrop`` — the *same* argmax selection, but dispatched
   through ``MoERouter(k=1)`` with a capacity large enough that nothing is
   dropped. Isolates the measurement path through ``MoERouter``; selection is
   identical to (1) so any loss delta is a bug, not a signal.
3. ``moerouter-topk-cap`` — ``MoERouter(k=2, capacity_factor=1.25)``: the real
   capacity-aware top-2 policy, where over-capacity assignments are *dropped*
   and each token's output is the (renormalized) mixture of its kept experts.

A dense ``NanoLM`` baseline, trained on the same data with ``hidden`` tuned so
its parameter count ≈ the MoE's *active* parameter count, gives the
matched-active-compute reference.

*What the study answers.* Does the production-grade routing policy
(capacity-aware top-k + Switch aux) measurably change load-balance and/or
held-out loss relative to the hand-rolled top-1 router, on identical experts
against a known floor? The verdict is a **measurement attributed to the floor**,
not a capability claim.

Honest scope: routing-policy ablation on fixed experts — a nano-scale
methodology study. NOT a claim that either router scales to frontier MoEs, and
NOT that a policy delta implies a capability delta. A "superior router" claim
needs the same measurement on a real model to the κ ≥ 0.40 / 2-judge gate.
The trainable Gumbel-softmax bridge (P7-B) is a separate, gated follow-up,
not included here. Pure stdlib + numpy.

    python -m pretraining.architecture.p7_router_ablation --quick
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:  # pragma: no cover
    _HAVE_NUMPY = False

from moe.router import MoERouter, load_balancing_loss  # noqa: E402
from pretraining.architecture.moe import MoELM  # noqa: E402
from pretraining.nano import (  # noqa: E402
    NanoLM, eval_loss, make_source, sample_stream, source_entropy, to_examples, train,
)

HERE = Path(__file__).resolve().parent

# Load-bearing phrase asserted by offline_invariants() AND the test (house rule).
SCOPE_KEY = "routing-policy ablation on fixed experts"


# ---------------------------------------------------------------------------
# Training (reuses MoELM.train_step; experts frozen afterward by copy)
# ---------------------------------------------------------------------------

def _train_moe(m: MoELM, examples, epochs: int, lr: float, seed: int) -> None:
    """Full-batch-per-epoch SGD over MoELM (matches run_sparse_quant._train_moe)."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for j in order:
            ctx, t = examples[j]
            m.train_step(ctx, t, lr)


def _eval_moe_native(m: MoELM, examples) -> float:
    """Mean NLL via MoELM's own forward — the self-consistency reference."""
    return sum(m.nll(c, t) for c, t in examples) / max(1, len(examples))


# ---------------------------------------------------------------------------
# Shared gating: compute router logits ONCE from MoELM's trained router.
# ---------------------------------------------------------------------------

def _router_logits_matrix(m: MoELM, examples) -> "tuple[np.ndarray, list[list[int]]]":
    """Return (T, E) router logits + the per-token active-input lists.

    Reproduces MoELM._route's logit accumulation (bias + sparse Wr rows) exactly,
    so argmax over these == MoELM's chosen expert. Computed once; every policy
    consumes this same array (the controlled-variable invariant).
    """
    E = m.n_experts
    T = len(examples)
    logits = np.zeros((T, E), dtype=np.float64)
    actives: list[list[int]] = []
    for t, (ctx, _tgt) in enumerate(examples):
        active = m._active_inputs(ctx)
        actives.append(active)
        row_logits = list(m.br)
        for p in active:
            wrow = m.Wr[p]
            for e in range(E):
                row_logits[e] += wrow[e]
        logits[t] = row_logits
    return logits, actives


# ---------------------------------------------------------------------------
# Per-policy evaluation. Each consumes the SAME logits matrix.
# ---------------------------------------------------------------------------

def _expert_output(m: MoELM, exp, active) -> list[float]:
    """Softmax output distribution of one (frozen) expert on one token."""
    _, probs = m._expert_forward(exp, active)
    return probs


def _eval_handrolled_top1(m: MoELM, logits, actives, examples):
    """Argmax → one expert (faithful replica of MoELM.forward). No capacity."""
    E = m.n_experts
    total_loss = 0.0
    counts = [0] * E
    for t, (_ctx, tgt) in enumerate(examples):
        # argmax over logits, lowest-index tiebreak (matches MoELM._route).
        choice = max(range(E), key=lambda e: logits[t, e])
        counts[choice] += 1
        probs = _expert_output(m, m.experts[choice], actives[t])
        total_loss += -math.log(max(probs[tgt], 1e-12))
    held = total_loss / max(1, len(examples))
    probs_full = _softmax_rows(logits)
    aux = load_balancing_loss(
        np.argmax(logits, axis=1).reshape(-1, 1), probs_full, E)
    return _policy_stats(m, held, aux, counts, dropped_slots=0, slots_attempted=len(examples))


def _eval_moerouter(m: MoELM, logits, actives, examples, *, k: int, capacity_factor: float):
    """Dispatch the shared logits through MoERouter; mix kept experts per token."""
    T = len(examples)
    router = MoERouter(m.n_experts, k=k, capacity_factor=capacity_factor, seed=0)
    plan = router.route(logits)
    # invert dispatch[e] = [(token, weight), ...] -> token -> [(expert, weight), kept?]
    per_token: list[list[tuple[int, float]]] = [[] for _ in range(T)]
    for e, items in plan["dispatch"].items():
        for (tok, w) in items:
            per_token[tok].append((e, w))

    total_loss = 0.0
    counts = list(plan["counts"])
    for t, (_ctx, tgt) in enumerate(examples):
        kept = per_token[t]
        if not kept:  # every slot dropped — honest "no expert fired" → uniform
            total_loss += math.log(m.V)
            continue
        wsum = sum(w for _e, w in kept)
        mix = [0.0] * m.V
        for e, w in kept:
            probs = _expert_output(m, m.experts[e], actives[t])
            wn = w / wsum  # renormalize kept weights → valid mixture distribution
            for k_idx in range(m.V):
                mix[k_idx] += wn * probs[k_idx]
        total_loss += -math.log(max(mix[tgt], 1e-12))
    held = total_loss / max(1, T)
    slots_attempted = T * k
    return _policy_stats(
        m, held, plan["aux_loss"], counts,
        dropped_slots=plan["dropped"], slots_attempted=slots_attempted)


def _policy_stats(m: MoELM, held: float, aux: float, counts, *,
                  dropped_slots: int, slots_attempted: int) -> dict:
    total = sum(counts) or 1
    max_share = max(counts) / total
    mean = (sum(counts) / len(counts)) if counts else 0.0
    var = sum((c - mean) ** 2 for c in counts) / max(1, len(counts))
    cv = math.sqrt(var) / mean if mean > 0 else float("inf")
    return {
        "held_loss": round(held, 5),
        "aux_loss": round(aux, 5),
        "max_route_share": round(max_share, 4),
        "route_cv": round(cv, 4),
        "counts": list(counts),
        "pct_dropped": round(100.0 * dropped_slots / max(1, slots_attempted), 3),
    }


# ---------------------------------------------------------------------------
# Dense baseline at matched ACTIVE compute (param count equalized, not just width)
# ---------------------------------------------------------------------------

def _dense_hidden_for_match(vocab: int, context: int, moe_active: int) -> int:
    """Solve for dense hidden whose param count ≈ the MoE's active param count.

    dense.params(h) = in_dim*h + h + h*V + V  →  h = (P - V) / (in_dim + 1 + V)
    """
    in_dim = context * vocab
    h = (moe_active - vocab) / (in_dim + 1 + vocab)
    return max(1, int(round(h)))


# ---------------------------------------------------------------------------
# Softmax helper (mirrors moe.router._softmax)
# ---------------------------------------------------------------------------

def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# The ablation
# ---------------------------------------------------------------------------

def run_ablation(*, quick: bool = False, seed: int = 0) -> dict:
    if not _HAVE_NUMPY:
        raise RuntimeError("numpy required for P7 router ablation")

    if quick:
        vocab, context, hidden, n_experts = 8, 2, 16, 4
        n_train, n_eval, epochs, lr, peak = 120, 60, 4, 0.1, 3.0
    else:
        vocab, context, hidden, n_experts = 10, 2, 32, 4
        n_train, n_eval, epochs, lr, peak = 300, 150, 8, 0.1, 3.0

    src = make_source(vocab=vocab, order=context, seed=seed, peak=peak)
    E = source_entropy(src)
    train_ex = to_examples(sample_stream(src, n_train, seed=seed), context=context)
    eval_ex = to_examples(sample_stream(src, n_eval, seed=seed + 777), context=context)

    # --- Train one MoE, then FREEZE experts (deepcopy isolates the frozen snapshot) ---
    moe_trained = MoELM(vocab, context, hidden, n_experts, seed=seed)
    _train_moe(moe_trained, train_ex, epochs, lr, seed=seed)
    m = copy.deepcopy(moe_trained)  # frozen substrate: routing passes must not mutate it

    # Snapshot experts for the frozen-experts invariant.
    experts_before = _expert_fingerprint(m)

    # --- Shared gating logits (computed ONCE; the controlled variable) ---
    logits, actives = _router_logits_matrix(m, eval_ex)

    # --- The three policies, all on the same frozen experts + same logits ---
    p_handrolled = _eval_handrolled_top1(m, logits, actives, eval_ex)
    p_top1_nodrop = _eval_moerouter(
        m, logits, actives, eval_ex, k=1, capacity_factor=64.0)
    p_topk_cap = _eval_moerouter(
        m, logits, actives, eval_ex, k=2, capacity_factor=1.25)

    experts_after = _expert_fingerprint(m)
    experts_frozen = (experts_before == experts_after)

    # --- Self-consistency: handrolled == MoELM.native (same argmax selection) ---
    native_held = _eval_moe_native(m, eval_ex)
    handrolled_matches_native = abs(p_handrolled["held_loss"] - round(native_held, 5)) < 1e-3

    # --- Dense baseline at matched active compute ---
    moe_active = m.active_params()
    h_dense = _dense_hidden_for_match(vocab, context, moe_active)
    dense = NanoLM(vocab, context, h_dense, seed=seed)
    train(dense, train_ex, epochs=epochs, optimizer="sgd", lr=lr, seed=seed)
    dense_held = eval_loss(dense, eval_ex)

    # --- Verdict: MoERouter top-k-cap vs handrolled top-1 (attributed to floor) ---
    gap = p_handrolled["held_loss"] - p_topk_cap["held_loss"]
    if gap > 1e-3:
        verdict = "moe_router_better"
    elif gap < -1e-3:
        verdict = "handrolled_better"
    else:
        verdict = "tie"

    # --- Decision note for the gated P7-B follow-up (documented, not auto-acted).
    # Honest about the confound: a top-k CAP loss advantage conflates two effects —
    # (a) the capacity-drop POLICY, and (b) plain MIXTURE-AVERAGING (k>1 experts
    # blended per token smooth the output distribution, which lowers NLL regardless
    # of *which* experts or whether dropping happened). We disentangle partly via
    # moerouter-top1-nodrop (same k=1, no mixture confound) but the top-2 row cannot
    # be attributed to policy alone. So "FOR B" requires a loss advantage that
    # survives acknowledging the mixture confound — i.e. we are conservative. ---
    loss_better = p_topk_cap["held_loss"] < p_handrolled["held_loss"] - 0.05
    balance_better = (
        p_topk_cap["max_route_share"] < p_handrolled["max_route_share"] - 1e-3
        or p_topk_cap["route_cv"] < p_handrolled["route_cv"] - 1e-3)
    if loss_better and balance_better:
        b_note = (
            "evidence FOR P7-B (provisional): top-k-cap held-out loss is lower AND load "
            "is better spread than the hand-rolled top-1 router. CAVEAT: the loss "
            "advantage may be partly a mixture-averaging confound (k>1 smooths the "
            "output), not the capacity policy — a trainable bridge is worth building "
            "only if the advantage holds after isolating the policy effect.")
    else:
        b_note = (
            "evidence AGAINST P7-B: no clear joint advantage on loss AND load balance "
            "at nano scale — heuristic router sufficient here. Advanced-router "
            "benefits, if any, appear at larger scale; do not build the bridge on this.")

    return {
        "study": "P7 — router-policy ablation on fixed nano-MoE experts",
        "floor_E": round(E, 5),
        "config": {
            "vocab": vocab, "context": context, "hidden": hidden,
            "n_experts": n_experts, "epochs": epochs, "seed": seed,
            "n_train": len(train_ex), "n_eval": len(eval_ex),
        },
        "moe_active_params": moe_active,
        "moe_total_params": m.num_params(),
        "dense": {
            "hidden_tuned": h_dense,
            "params": dense.num_params(),
            "held_loss": round(dense_held, 5),
            "excess_over_floor": round(dense_held - E, 5),
        },
        "compute_equalized": abs(dense.num_params() - moe_active) / max(1, moe_active) < 0.05,
        "policies": {
            "handrolled-top1": _with_floor(p_handrolled, E),
            "moerouter-top1-nodrop": _with_floor(p_top1_nodrop, E),
            "moerouter-topk-cap": _with_floor(p_topk_cap, E),
        },
        "self_consistency": {
            "native_moe_held_loss": round(native_held, 5),
            "handrolled_matches_native": handrolled_matches_native,
            "experts_frozen": experts_frozen,
        },
        "verdict": verdict,
        "held_loss_gap_handrolled_minus_topkcap": round(gap, 5),
        "p7b_decision_note": b_note,
        "honest_scope": (
            "Routing-policy ablation on fixed experts — a nano-scale methodology study. "
            "Measures whether capacity-aware top-k routing (MoERouter + Switch aux loss) "
            "gives different load-balance / held-out loss than MoELM's hand-rolled top-1 "
            "router, on identical frozen experts against a known entropy floor. NOT a claim "
            "that either router scales to frontier MoEs, and NOT that the policy delta "
            "implies a capability delta. A 'superior router' claim needs the same "
            "measurement on a real model to the kappa >= 0.40 / 2-judge gate. The trainable "
            "Gumbel-softmax bridge (P7-B) is a separate, gated follow-up, not included here."
        ),
    }


def _with_floor(policy: dict, E: float) -> dict:
    out = dict(policy)
    out["excess_over_floor"] = round(policy["held_loss"] - E, 5)
    return out


def _expert_fingerprint(m: MoELM):
    """Stable summary of expert weights — equal iff experts unchanged."""
    fp = []
    for exp in m.experts:
        w1 = sum(sum(row) for row in exp["W1"])
        w2 = sum(sum(row) for row in exp["W2"])
        fp.append((round(w1, 12), round(w2, 12)))
    return tuple(fp)


# ---------------------------------------------------------------------------
# Offline invariants (house contract: tuple[bool, dict] with {"checks": ..., **detail})
# ---------------------------------------------------------------------------

def offline_invariants() -> "tuple[bool, dict]":
    if not _HAVE_NUMPY:
        return False, {"checks": {"numpy_available": False}}
    checks: dict[str, bool] = {}
    detail: dict = {}

    rep = run_ablation(quick=True, seed=0)

    # 1. Experts frozen across all routing passes.
    checks["experts_frozen"] = bool(rep["self_consistency"]["experts_frozen"])

    # 2. Shared gating isolated: handrolled == native MoELM (same argmax selection
    #    from the shared logits → identical loss). Any delta here means the gating
    #    was NOT shared or the replica diverged.
    checks["shared_gating_isolated"] = bool(rep["self_consistency"]["handrolled_matches_native"])

    # 3. Compute equalized: dense param count within 5% of MoE active params.
    dense_p = rep["dense"]["params"]
    moe_a = rep["moe_active_params"]
    detail["dense_params"] = dense_p
    detail["moe_active_params"] = moe_a
    checks["compute_equalized"] = abs(dense_p - moe_a) / max(1, moe_a) < 0.05

    # 4. moerouter-top1-nodrop == handrolled (selection identical when nothing drops):
    #    the only legitimate difference between them is zero.
    h = rep["policies"]["handrolled-top1"]["held_loss"]
    n = rep["policies"]["moerouter-top1-nodrop"]["held_loss"]
    detail["nodrop_vs_handrolled_delta"] = round(abs(h - n), 6)
    checks["nodrop_matches_handrolled"] = abs(h - n) < 1e-3

    # 5. aux loss floor: forced-uniform routing → aux == 1.0; full skew → aux > 1.0.
    uniform = np.zeros((48, 4))
    p_u = _softmax_rows(uniform)
    idx_u = (np.arange(48).reshape(-1, 1)) % 4
    aux_uniform = load_balancing_loss(idx_u, p_u, 4)
    skew = np.zeros((48, 4)); skew[:, 0] = 12.0
    p_s = _softmax_rows(skew)
    idx_s = np.argmax(skew, axis=1).reshape(-1, 1)
    aux_skew = load_balancing_loss(idx_s, p_s, 4)
    detail["aux_uniform"] = round(aux_uniform, 6)
    detail["aux_skew"] = round(aux_skew, 4)
    checks["aux_loss_floor_respected"] = (
        abs(aux_uniform - 1.0) < 1e-9 and aux_skew > 1.0 + 1e-6)

    # 6. Determinism: two runs, same seed → identical rounded numbers.
    rep2 = run_ablation(quick=True, seed=0)
    checks["deterministic"] = (
        rep["policies"] == rep2["policies"]
        and rep["verdict"] == rep2["verdict"]
        and rep["floor_E"] == rep2["floor_E"])

    # 7. Honest scope present with the load-bearing phrase (case-insensitive:
    #    the scope string leads the sentence, so it is capitalized in prose).
    checks["scope_present"] = SCOPE_KEY.lower() in rep["honest_scope"].lower()

    # 8. All reported losses finite and the floor is positive.
    losses = [rep["floor_E"], rep["dense"]["held_loss"]]
    losses += [p["held_loss"] for p in rep["policies"].values()]
    checks["losses_finite_floor_positive"] = all(
        math.isfinite(x) for x in losses) and rep["floor_E"] > 0

    ok = all(checks.values())
    return ok, {"checks": checks, **detail}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rep = run_ablation(quick=args.quick, seed=args.seed)
    out = args.out or (HERE / ("p7-router-ablation-quick-latest.json"
                                if args.quick else "p7-router-ablation-latest.json"))
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")

    E = rep["floor_E"]
    print(f"floor E = {E}")
    print(f"MoE: active={rep['moe_active_params']} total={rep['moe_total_params']}")
    print(f"dense (matched active): hidden={rep['dense']['hidden_tuned']} "
          f"params={rep['dense']['params']} held={rep['dense']['held_loss']}")
    for name, p in rep["policies"].items():
        print(f"  {name:24s} held={p['held_loss']:7.4f} "
              f"aux={p['aux_loss']:6.3f} max_share={p['max_route_share']:.3f} "
              f"cv={p['route_cv']:.3f} dropped={p['pct_dropped']}%")
    print(f"verdict: {rep['verdict']}  "
          f"(gap handrolled-topkcap={rep['held_loss_gap_handrolled_minus_topkcap']})")
    print(f"P7-B: {rep['p7b_decision_note']}")


if __name__ == "__main__":
    main()

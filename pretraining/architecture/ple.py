# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Per-Layer Embeddings (PLE) at nano scale — lookup-heavy capacity at matched compute.

The externally-claimed "intelligence-per-byte" architecture idea (PLE, as shipped in
Gemma 3n's edge variants — NOT a mythical "Gemma 4") is: give each layer its own small
per-token embedding added into the residual/hidden stream. Those parameters are pure
LOOKUPS — only the active token's row participates per forward — so they add representational
capacity WITHOUT inflating the compute-heavy matmul. The honest, falsifiable version of that
claim, at nano scale against a known entropy floor:

    Does adding a per-token embedding table to the hidden pre-activation lower held loss
    versus a dense baseline of the SAME hidden width (== the same compute-bound h×V matmul)?

``PLELM`` is ``NanoLM`` plus a shared per-token embedding ``E[token] -> R^h`` summed into the
hidden pre-activation. Same ``h`` => identical active matmul FLOPs (h×V); the PLE table adds
``V×h`` total params but only ``c`` extra lookup rows per token. Real hand-written backprop
(numerically gradient-checked in ``offline_invariants``). Pure stdlib; a reference probe, NOT
a SOTA or deployment claim — see ARCHITECTURE.md for the real design.
"""
from __future__ import annotations

import math
import random
from typing import Any

from pretraining.nano.model import NanoLM


class PLELM(NanoLM):
    """NanoLM augmented with a shared per-token Per-Layer Embedding into the hidden layer.

    ``ple=False`` (or a zero-initialised table) makes it behave IDENTICALLY to ``NanoLM`` —
    the clean self-consistency baseline. With ``ple=True`` the table is small-random init and
    trains alongside the rest by hand-written backprop.
    """

    def __init__(self, vocab: int, context: int, hidden: int, *, seed: int = 0,
                 ple: bool = True) -> None:
        super().__init__(vocab, context, hidden, seed=seed)
        self.ple = ple
        rng = random.Random(seed + 7)
        s = (1.0 / math.sqrt(max(1, hidden))) if ple else 0.0
        # E[token] -> R^h, shared across context positions (a per-LAYER, per-token embedding).
        self.E = [[rng.uniform(-s, s) for _ in range(hidden)] for _ in range(vocab)]

    # -- parameter / compute accounting --------------------------------------
    def num_params(self) -> int:
        return super().num_params() + self.V * self.h

    def ple_params(self) -> int:
        return self.V * self.h

    def active_matmul_flops(self) -> int:
        """The compute-bound term (hidden->vocab matmul), IDENTICAL to a dense NanoLM of the
        same hidden width. This is the 'matched compute' axis the PLE claim rests on."""
        return self.h * self.V

    def active_lookup_rows(self) -> int:
        """Embedding rows touched per token: the c W1 rows + the c PLE rows (cheap adds)."""
        return self.c + (self.c if self.ple else 0)

    # -- forward --------------------------------------------------------------
    def _ple_sum(self, ctx: "list[int]") -> "list[float]":
        acc = [0.0] * self.h
        if not self.ple:
            return acc
        for tok in ctx:
            row = self.E[tok]
            for i in range(self.h):
                acc[i] += row[i]
        return acc

    def forward(self, ctx: "list[int]") -> "tuple[list[float], list[float]]":
        active = self._active_inputs(ctx)
        pre1 = list(self.b1)
        for p in active:
            row = self.W1[p]
            for i in range(self.h):
                pre1[i] += row[i]
        ple = self._ple_sum(ctx)
        for i in range(self.h):
            pre1[i] += ple[i]
        a1 = [math.tanh(x) for x in pre1]
        logits = list(self.b2)
        for i in range(self.h):
            ai = a1[i]
            w2i = self.W2[i]
            for k in range(self.V):
                logits[k] += ai * w2i[k]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        z = sum(exps)
        return a1, [e / z for e in exps]

    # -- single-example gradient + SGD step ----------------------------------
    def train_step(self, ctx: "list[int]", target: int, lr: float) -> float:
        active = self._active_inputs(ctx)
        a1, probs = self.forward(ctx)
        loss = -math.log(max(probs[target], 1e-12))

        dlogits = list(probs)
        dlogits[target] -= 1.0
        # W2, b2
        for i in range(self.h):
            ai = a1[i]
            w2i = self.W2[i]
            for k in range(self.V):
                w2i[k] -= lr * (ai * dlogits[k])
        da1 = [0.0] * self.h
        for i in range(self.h):
            w2i = self.W2[i]
            s = 0.0
            for k in range(self.V):
                s += w2i[k] * dlogits[k]
            da1[i] = s
        for k in range(self.V):
            self.b2[k] -= lr * dlogits[k]
        dpre1 = [da1[i] * (1.0 - a1[i] * a1[i]) for i in range(self.h)]
        # b1, W1 (active rows only)
        for i in range(self.h):
            self.b1[i] -= lr * dpre1[i]
        for p in active:
            row = self.W1[p]
            for i in range(self.h):
                row[i] -= lr * dpre1[i]
        # PLE table: each context token's row gets dpre1 (a token may repeat -> accumulates).
        if self.ple:
            for tok in ctx:
                row = self.E[tok]
                for i in range(self.h):
                    row[i] -= lr * dpre1[i]
        return loss


def _train(model: PLELM, examples, epochs: int, lr: float, seed: int) -> None:
    rng = random.Random(seed)
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for j in order:
            ctx, t = examples[j]
            model.train_step(ctx, t, lr)


def _eval(model: PLELM, examples) -> float:
    return sum(model.nll(c, t) for c, t in examples) / max(1, len(examples))


def run(*, quick: bool = False, out: "Any | None" = None) -> dict:
    """Dense NanoLM vs PLELM at matched hidden width (== matched active matmul)."""
    import json
    from pathlib import Path

    from pretraining.nano import make_source, sample_stream, source_entropy, to_examples
    from pretraining.nano.train import train

    vocab, order, context, hidden = 8, 2, 2, 8
    epochs = 8 if quick else 14
    src = make_source(vocab=vocab, order=order, seed=1, peak=3.0)
    E = source_entropy(src)
    ex = to_examples(sample_stream(src, 2000 + context, seed=2), context=context)
    held = to_examples(sample_stream(src, 1200, seed=99), context=context)

    dense = NanoLM(vocab=vocab, context=context, hidden=hidden, seed=0)
    train(dense, ex, epochs=epochs, optimizer="sgd", lr=0.1, seed=0)
    dense_loss = eval_held(dense, held)

    ple = PLELM(vocab=vocab, context=context, hidden=hidden, seed=0, ple=True)
    _train(ple, ex, epochs, 0.1, 0)
    ple_loss = _eval(ple, held)

    report = {
        "study": "architecture probe — Per-Layer Embeddings vs dense at matched compute (nano LM)",
        "honesty_note": ("Toy. Tests whether a per-token embedding table lowers held loss at the "
                         "SAME hidden width (== same h×V matmul). PLE is a Gemma 3n feature, not a "
                         "'Gemma 4'; this is a falsifiable nano probe, not a SOTA/byte claim."),
        "analytic_floor_E": round(E, 5),
        "config": {"vocab": vocab, "context": context, "hidden": hidden, "epochs": epochs},
        "dense": {"total_params": dense.num_params(),
                  "active_matmul_flops": hidden * vocab,
                  "held_loss": round(dense_loss, 5),
                  "excess_over_floor": round(dense_loss - E, 5)},
        "ple": {"total_params": ple.num_params(),
                "ple_params": ple.ple_params(),
                "active_matmul_flops": ple.active_matmul_flops(),  # identical to dense
                "active_lookup_rows": ple.active_lookup_rows(),
                "held_loss": round(ple_loss, 5),
                "excess_over_floor": round(ple_loss - E, 5)},
        "matched_compute": dense.num_params() and (hidden * vocab == ple.active_matmul_flops()),
        "verdict": ("ple_better" if ple_loss < dense_loss - 1e-3
                    else "dense_better" if dense_loss < ple_loss - 1e-3 else "tie"),
    }
    out = Path(out) if out else (Path(__file__).resolve().parent / "ple-probe-latest.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def eval_held(model, examples) -> float:
    from pretraining.nano.model import eval_loss
    return eval_loss(model, examples)


def offline_invariants() -> "tuple[bool, dict]":
    """Reference invariants: PLE-off == dense, param accounting, matched compute, and a
    finite-difference gradient check on the embedding table (backprop correctness)."""
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    vocab, context, hidden = 6, 2, 5

    # 1. PLE-off behaves EXACTLY like the dense NanoLM (same seed) -> identical probs.
    base = NanoLM(vocab=vocab, context=context, hidden=hidden, seed=0)
    off = PLELM(vocab=vocab, context=context, hidden=hidden, seed=0, ple=False)
    _, pb = base.forward([1, 2])
    _, po = off.forward([1, 2])
    checks["ple_off_equals_dense"] = all(abs(a - b) < 1e-12 for a, b in zip(pb, po))

    # 2. Parameter accounting: PLE adds exactly V*h params; matmul FLOPs unchanged.
    on = PLELM(vocab=vocab, context=context, hidden=hidden, seed=0, ple=True)
    checks["ple_adds_V_times_h_params"] = on.num_params() == base.num_params() + vocab * hidden
    checks["matched_active_matmul"] = on.active_matmul_flops() == hidden * vocab
    detail["dense_params"] = base.num_params()
    detail["ple_params"] = on.num_params()

    # 3. forward is a valid distribution.
    _, probs = on.forward([0, 3])
    checks["forward_normalised"] = abs(sum(probs) - 1.0) < 1e-9

    # 4. Finite-difference gradient check on the embedding row of an active token.
    #    Analytic dE[tok][i] for one example == numeric (loss(E+eps)-loss(E-eps))/2eps.
    m = PLELM(vocab=vocab, context=context, hidden=hidden, seed=3, ple=True)
    ctx, target = [2, 4], 1
    a1, probs = m.forward(ctx)
    dlogits = list(probs); dlogits[target] -= 1.0
    da1 = [sum(m.W2[i][k] * dlogits[k] for k in range(vocab)) for i in range(hidden)]
    dpre1 = [da1[i] * (1.0 - a1[i] * a1[i]) for i in range(hidden)]
    # token 2 appears once in ctx -> analytic grad on E[2][i] == dpre1[i]
    tok, i = 2, 0
    analytic = dpre1[i]
    eps = 1e-5
    m.E[tok][i] += eps
    lp = m.nll(ctx, target)
    m.E[tok][i] -= 2 * eps
    lm = m.nll(ctx, target)
    m.E[tok][i] += eps  # restore
    numeric = (lp - lm) / (2 * eps)
    detail["grad_analytic"] = round(analytic, 6)
    detail["grad_numeric"] = round(numeric, 6)
    checks["embedding_gradcheck"] = abs(analytic - numeric) < 1e-4

    # 5. PLE actually trains: a few steps lower the loss on a tiny stream.
    m2 = PLELM(vocab=vocab, context=context, hidden=hidden, seed=1, ple=True)
    ex = [([0, 1], 2), ([1, 2], 3), ([2, 3], 4)]
    before = _eval(m2, ex)
    _train(m2, ex, epochs=40, lr=0.2, seed=0)
    after = _eval(m2, ex)
    checks["ple_trains_lowers_loss"] = after < before
    detail["train_loss_before"] = round(before, 5)
    detail["train_loss_after"] = round(after, 5)

    return all(checks.values()), {"checks": checks, **detail}


def main() -> None:
    # Run as a module so the `pretraining` package resolves: python -m pretraining.architecture.ple
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="fewer epochs")
    ap.add_argument("--invariants", action="store_true", help="run the offline reference checks instead of the probe")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.invariants:
        ok, detail = offline_invariants()
        print("PLE offline invariants:", "PASS" if ok else "FAIL")
        for k, v in detail["checks"].items():
            print(f"  [{'ok' if v else 'XX'}] {k}")
        raise SystemExit(0 if ok else 1)
    r = run(quick=args.quick, out=args.out)
    print(f"floor E = {r['analytic_floor_E']}")
    print(f"dense: held={r['dense']['held_loss']} params={r['dense']['total_params']} "
          f"matmul={r['dense']['active_matmul_flops']}")
    print(f"ple  : held={r['ple']['held_loss']} params={r['ple']['total_params']} "
          f"matmul={r['ple']['active_matmul_flops']} (matched={r['matched_compute']})")
    print(f"verdict: {r['verdict']}")


if __name__ == "__main__":
    main()

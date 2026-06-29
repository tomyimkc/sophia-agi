# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A genuinely-trainable nano language model in pure Python (no numpy/torch).

This is the shared substrate for every pretraining-research study in ``pretraining/``.
It is small on purpose — the contribution is *honest methodology* (real gradients,
real measured loss, a verifiable irreducible-loss floor), not frontier scale.

Architecture: fixed context window of ``c`` previous symbols, each one-hot, concatenated
into the input layer; one ``tanh`` hidden layer of width ``h``; softmax over the vocab.
Trained by hand-written backprop. Parameter count scales with ``h`` (capacity) so the
model supports clean ``L(N)`` capacity-scaling and ``L(D)`` data-scaling sweeps.

Why this matters for a scaling-law study: the training corpus (``data.py``) is an
order-``k`` Markov source whose *true* conditional entropy is computable in closed form.
That entropy is the irreducible loss ``E``. A fitted power law ``L = E + A·x^-p`` can
therefore be checked against ground truth — the floor is known, not guessed.

    from pretraining.nano.model import NanoLM
    from pretraining.nano.train import train
    m = NanoLM(vocab=16, context=2, hidden=32, seed=0)
    hist = train(m, corpus, epochs=8)   # real cross-entropy descent
"""
from __future__ import annotations

import math
import random
from typing import Any


class NanoLM:
    """One-hidden-layer softmax language model with hand-written backprop."""

    def __init__(self, vocab: int, context: int, hidden: int, *, seed: int = 0) -> None:
        self.V = vocab
        self.c = context
        self.h = hidden
        self.in_dim = context * vocab
        rng = random.Random(seed)
        # Small symmetric init; scale by fan-in to keep early activations tame.
        s1 = 1.0 / math.sqrt(max(1, self.in_dim))
        s2 = 1.0 / math.sqrt(max(1, hidden))
        self.W1 = [[rng.uniform(-s1, s1) for _ in range(hidden)] for _ in range(self.in_dim)]
        self.b1 = [0.0 for _ in range(hidden)]
        self.W2 = [[rng.uniform(-s2, s2) for _ in range(vocab)] for _ in range(hidden)]
        self.b2 = [0.0 for _ in range(vocab)]

    # -- parameter accounting -------------------------------------------------
    def num_params(self) -> int:
        """Total trainable scalars (the ``N`` axis of the scaling law)."""
        return self.in_dim * self.h + self.h + self.h * self.V + self.V

    # -- forward --------------------------------------------------------------
    def _active_inputs(self, ctx: "list[int]") -> "list[int]":
        """Flat indices set to 1 by the concatenated one-hot context."""
        return [j * self.V + ctx[j] for j in range(self.c)]

    def forward(self, ctx: "list[int]") -> "tuple[list[float], list[float]]":
        """Return (hidden activations a1, softmax probabilities)."""
        active = self._active_inputs(ctx)
        pre1 = list(self.b1)
        for p in active:
            row = self.W1[p]
            for i in range(self.h):
                pre1[i] += row[i]
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
        probs = [e / z for e in exps]
        return a1, probs

    def nll(self, ctx: "list[int]", target: int) -> float:
        """Per-token negative log-likelihood (natural log = nats)."""
        _, probs = self.forward(ctx)
        return -math.log(max(probs[target], 1e-12))

    # -- single-example gradient ---------------------------------------------
    def grads(self, ctx: "list[int]", target: int) -> "tuple[dict, float]":
        """Hand-written backprop. Returns (grad dict, loss). Only the active input
        rows of W1 receive gradient, which keeps the backward pass cheap."""
        active = self._active_inputs(ctx)
        a1, probs = self.forward(ctx)
        loss = -math.log(max(probs[target], 1e-12))

        dlogits = list(probs)
        dlogits[target] -= 1.0

        db2 = dlogits
        dW2 = [[a1[i] * dlogits[k] for k in range(self.V)] for i in range(self.h)]

        da1 = [0.0] * self.h
        for i in range(self.h):
            w2i = self.W2[i]
            s = 0.0
            for k in range(self.V):
                s += w2i[k] * dlogits[k]
            da1[i] = s
        dpre1 = [da1[i] * (1.0 - a1[i] * a1[i]) for i in range(self.h)]
        db1 = dpre1
        dW1 = {p: list(dpre1) for p in active}   # sparse: only active rows
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}, loss


def eval_loss(model: NanoLM, examples: "list[tuple[list[int], int]]") -> float:
    """Mean per-token NLL (nats) over held-out (context, target) pairs."""
    if not examples:
        return float("nan")
    total = 0.0
    for ctx, t in examples:
        total += model.nll(ctx, t)
    return total / len(examples)


__all__ = ["NanoLM", "eval_loss"]

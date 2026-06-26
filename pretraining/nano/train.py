# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Training loop + pluggable optimizers for the nano LM (pure Python).

Optimizers: ``sgd``, ``momentum``, ``adam`` — enough to study training dynamics and
stability (the algorithm direction's "高性能和鲁棒的优化器…动力学和稳定性" line) without
a deep-learning framework. Each step logs the global gradient norm so the optimizer-probe
study can compare convergence *and* stability (spike count, divergence) across optimizers.

    from pretraining.nano.train import train
    hist = train(model, examples, epochs=8, optimizer="adam", lr=0.05)
    hist["epoch_loss"]   # mean training NLL per epoch (nats)
    hist["grad_norms"]   # per-step global grad norm
"""
from __future__ import annotations

import math
import random
from typing import Any

from pretraining.nano.model import NanoLM


class _State:
    """Per-optimizer accumulators (momentum / Adam moments), created lazily."""

    def __init__(self) -> None:
        self.m: dict = {}
        self.v: dict = {}
        self.t = 0


def _grad_global_norm(g: dict) -> float:
    total = 0.0
    for p, row in g["W1"].items():
        for x in row:
            total += x * x
    for x in g["b1"]:
        total += x * x
    for row in g["W2"]:
        for x in row:
            total += x * x
    for x in g["b2"]:
        total += x * x
    return math.sqrt(total)


def _apply(model: NanoLM, g: dict, opt: str, lr: float, st: _State,
           *, beta1: float = 0.9, beta2: float = 0.999, mu: float = 0.9,
           eps: float = 1e-8) -> None:
    """Apply one parameter update. Sparse on W1 (only active rows have gradient)."""
    st.t += 1

    def upd(name: str, ref, grad, idx) -> float:
        """Return the new value for one scalar parameter at key (name, idx)."""
        if opt == "sgd":
            return ref - lr * grad
        key = (name, idx)
        if opt == "momentum":
            m = st.m.get(key, 0.0)
            m = mu * m + grad
            st.m[key] = m
            return ref - lr * m
        # adam
        m = st.m.get(key, 0.0)
        v = st.v.get(key, 0.0)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        st.m[key] = m
        st.v[key] = v
        mhat = m / (1 - beta1 ** st.t)
        vhat = v / (1 - beta2 ** st.t)
        return ref - lr * mhat / (math.sqrt(vhat) + eps)

    for p, row in g["W1"].items():
        w = model.W1[p]
        for i, gi in enumerate(row):
            w[i] = upd("W1", w[i], gi, (p, i))
    for i, gi in enumerate(g["b1"]):
        model.b1[i] = upd("b1", model.b1[i], gi, i)
    for i, row in enumerate(g["W2"]):
        w = model.W2[i]
        for k, gi in enumerate(row):
            w[k] = upd("W2", w[k], gi, (i, k))
    for k, gi in enumerate(g["b2"]):
        model.b2[k] = upd("b2", model.b2[k], gi, k)


def train(model: NanoLM, examples: "list[tuple[list[int], int]]", *,
          epochs: int = 8, optimizer: str = "adam", lr: float = 0.05,
          seed: int = 0, clip: float | None = None) -> "dict[str, Any]":
    """Train in place. Returns history: per-epoch mean loss, per-step grad norms,
    and stability stats (max grad norm, NaN/divergence flag)."""
    rng = random.Random(seed)
    st = _State()
    epoch_loss: list[float] = []
    grad_norms: list[float] = []
    diverged = False
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        total = 0.0
        for j in order:
            ctx, t = examples[j]
            g, loss = model.grads(ctx, t)
            gn = _grad_global_norm(g)
            grad_norms.append(gn)
            if math.isnan(loss) or math.isinf(loss) or gn > 1e6:
                diverged = True
            if clip and gn > clip and gn > 0:
                scale = clip / gn
                for p in g["W1"]:
                    g["W1"][p] = [x * scale for x in g["W1"][p]]
                g["b1"] = [x * scale for x in g["b1"]]
                g["W2"] = [[x * scale for x in r] for r in g["W2"]]
                g["b2"] = [x * scale for x in g["b2"]]
            _apply(model, g, optimizer, lr, st)
            total += loss
        epoch_loss.append(total / max(1, len(examples)))
    return {
        "epoch_loss": epoch_loss,
        "final_train_loss": epoch_loss[-1] if epoch_loss else float("nan"),
        "grad_norms": grad_norms,
        "max_grad_norm": max(grad_norms) if grad_norms else 0.0,
        "diverged": diverged,
        "optimizer": optimizer,
        "lr": lr,
        "params": model.num_params(),
    }


__all__ = ["train"]

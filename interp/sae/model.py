# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""From-scratch TopK sparse autoencoder — pure-stdlib reference (no torch/numpy).

Why pure stdlib: this is the OFFLINE, CI-green reference SAE. Tiny tensors,
deterministic seeds, no GPU, no downloads — so the dictionary-learning objective
is unit-testable everywhere and we *demonstrably understand* it (mirroring
`agent/steering`'s pure-stdlib core). The PRODUCTION trainer (roadmap M2) uses
SAELens / a torch port on RunPod / the DGX Spark over harvested Qwen2.5-7B
activations; see requirements-interp.txt. This file is intentionally not that.

Method (cited): TopK activation + group sparsity — Gao et al., *Scaling and
Evaluating Sparse Autoencoders* (OpenAI, 2024). Unit-norm decoder columns and
dead-feature resampling — Bricken et al., *Towards Monosemanticity* (Anthropic,
2023). The TopK gate directly fixes L0 (= k), avoiding the L1-shrinkage bias.

Shapes: x ∈ R^{d_in} (a residual-stream activation), code a ∈ R^{d_hidden}
(d_hidden ≫ d_in, overcomplete). Decoder columns W_dec[:, f] are the dictionary
features, kept unit-norm.
"""
from __future__ import annotations

import math
import random


class TopKSAE:
    """Minimal TopK SAE with a plain SGD trainer over python lists.

    a = TopK_k(ReLU(W_enc x + b_enc));  x_hat = W_dec a + b_dec.
    Decoder columns are renormalized to unit L2 after every step.
    """

    def __init__(self, d_in: int, d_hidden: int, k: int, *, seed: int = 0) -> None:
        if k < 1 or k > d_hidden:
            raise ValueError(f"k={k} must be in [1, d_hidden={d_hidden}]")
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.k = k
        rng = random.Random(seed)
        sd = 1.0 / math.sqrt(d_in)
        # W_enc: d_hidden × d_in
        self.W_enc = [[rng.gauss(0.0, sd) for _ in range(d_in)] for _ in range(d_hidden)]
        self.b_enc = [0.0] * d_hidden
        # W_dec: d_in × d_hidden  (column f = feature f); init random, unit-norm cols
        self.W_dec = [[rng.gauss(0.0, sd) for _ in range(d_hidden)] for _ in range(d_in)]
        self.b_dec = [0.0] * d_in
        self._normalize_decoder()

    # --- forward ---------------------------------------------------------
    def encode(self, x: "list[float]") -> "tuple[list[float], list[int]]":
        """Return (code a, sorted list of active feature indices). |active| ≤ k."""
        z = self.b_enc[:]
        for h in range(self.d_hidden):
            row = self.W_enc[h]
            z[h] = self.b_enc[h] + sum(row[i] * x[i] for i in range(self.d_in))
        relu = [v if v > 0.0 else 0.0 for v in z]
        # TopK by magnitude over positive pre-activations.
        order = sorted(range(self.d_hidden), key=lambda h: relu[h], reverse=True)
        keep = sorted(h for h in order[: self.k] if relu[h] > 0.0)
        a = [0.0] * self.d_hidden
        for h in keep:
            a[h] = relu[h]
        return a, keep

    def decode(self, a: "list[float]") -> "list[float]":
        out = self.b_dec[:]
        for i in range(self.d_in):
            row = self.W_dec[i]
            out[i] = self.b_dec[i] + sum(row[h] * a[h] for h in range(self.d_hidden))
        return out

    def reconstruct(self, x: "list[float]") -> "list[float]":
        a, _ = self.encode(x)
        return self.decode(a)

    def encode_batch(self, X: "list[list[float]]") -> "list[list[float]]":
        return [self.encode(x)[0] for x in X]

    def reconstruct_batch(self, X: "list[list[float]]") -> "list[list[float]]":
        return [self.reconstruct(x) for x in X]

    # --- training --------------------------------------------------------
    def train_step(self, X: "list[list[float]]", *, lr: float = 0.5) -> float:
        """One full-batch SGD step on mean reconstruction MSE. Returns the loss
        (mean per-sample squared error) measured BEFORE the update."""
        n = len(X)
        if n == 0:
            return 0.0
        di, dh = self.d_in, self.d_hidden
        gWenc = [[0.0] * di for _ in range(dh)]
        gbenc = [0.0] * dh
        gWdec = [[0.0] * dh for _ in range(di)]
        gbdec = [0.0] * di
        total = 0.0
        for x in X:
            a, keep = self.encode(x)
            xhat = self.decode(a)
            # g = dL/dxhat for L = ||xhat - x||^2  (factor 2 folded in)
            g = [2.0 * (xhat[i] - x[i]) for i in range(di)]
            total += sum((xhat[i] - x[i]) ** 2 for i in range(di))
            # decoder grads (only active features contribute)
            for i in range(di):
                gi = g[i]
                gbdec[i] += gi
                row = gWdec[i]
                Wi = self.W_dec[i]
                for h in keep:
                    row[h] += gi * a[h]
            # backprop into the code, then through ReLU+TopK (mask=1 on active) → encoder
            for h in keep:
                da_h = sum(self.W_dec[i][h] * g[i] for i in range(di))
                gbenc[h] += da_h
                ge = gWenc[h]
                for i in range(di):
                    ge[i] += da_h * x[i]
        scale = lr / n
        for h in range(dh):
            self.b_enc[h] -= scale * gbenc[h]
            We, Ge = self.W_enc[h], gWenc[h]
            for i in range(di):
                We[i] -= scale * Ge[i]
        for i in range(di):
            self.b_dec[i] -= scale * gbdec[i]
            Wd, Gd = self.W_dec[i], gWdec[i]
            for h in range(dh):
                Wd[h] -= scale * Gd[h]
        self._normalize_decoder()
        return total / n

    def fit(self, X: "list[list[float]]", *, steps: int = 600, lr: float = 0.5) -> "list[float]":
        """Train for `steps` full-batch steps; return the loss curve."""
        return [self.train_step(X, lr=lr) for _ in range(steps)]

    # --- diagnostics -----------------------------------------------------
    def decoder_norms(self) -> "list[float]":
        return [
            math.sqrt(sum(self.W_dec[i][h] ** 2 for i in range(self.d_in)))
            for h in range(self.d_hidden)
        ]

    def _normalize_decoder(self) -> None:
        for h in range(self.d_hidden):
            norm = math.sqrt(sum(self.W_dec[i][h] ** 2 for i in range(self.d_in)))
            if norm > 1e-12:
                for i in range(self.d_in):
                    self.W_dec[i][h] /= norm

    def dead_features(self, X: "list[list[float]]") -> "list[int]":
        """Indices of features that never activate over X (Bricken resampling target)."""
        live = set()
        for x in X:
            _, keep = self.encode(x)
            live.update(keep)
        return [h for h in range(self.d_hidden) if h not in live]

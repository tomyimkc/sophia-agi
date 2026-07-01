# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A minimal top-1 mixture-of-experts hidden layer for the nano LM (pure Python).

DeepSeek's signature pretraining-architecture contributions are MLA (multi-head latent
attention) and fine-grained MoE with shared + routed experts. This is a faithful *toy*
of the MoE idea: replace the single dense hidden layer with ``n_experts`` hidden blocks
and a learned router that sends each token to its top-1 expert. At inference only one
expert fires, so active params per token stay ~constant while total capacity grows — the
sparse-scaling argument, demonstrated at nano scale with real gradients.

The router is trained with a straight-through-style update: only the chosen expert (and
the router logit for it) receive gradient. A load-balancing penalty discourages routing
collapse (all tokens to one expert) — the same failure MoE training fights at scale.

This is a reference implementation to study routing behavior, NOT a performance claim.
"""
from __future__ import annotations

import math
import random


class MoELM:
    """One-hidden-layer LM whose hidden block is a top-1 MoE over ``n_experts``."""

    def __init__(self, vocab: int, context: int, hidden: int, n_experts: int,
                 *, seed: int = 0, balance_coef: float = 0.01) -> None:
        self.V = vocab
        self.c = context
        self.h = hidden
        self.n_experts = n_experts
        self.in_dim = context * vocab
        self.balance_coef = balance_coef
        rng = random.Random(seed)
        s1 = 1.0 / math.sqrt(max(1, self.in_dim))
        s2 = 1.0 / math.sqrt(max(1, hidden))
        sr = 1.0 / math.sqrt(max(1, self.in_dim))
        # one (W1,b1,W2,b2) block per expert
        self.experts = []
        for _ in range(n_experts):
            self.experts.append({
                "W1": [[rng.uniform(-s1, s1) for _ in range(hidden)] for _ in range(self.in_dim)],
                "b1": [0.0] * hidden,
                "W2": [[rng.uniform(-s2, s2) for _ in range(vocab)] for _ in range(hidden)],
                "b2": [0.0] * vocab,
            })
        # router: input -> expert logits
        self.Wr = [[rng.uniform(-sr, sr) for _ in range(n_experts)] for _ in range(self.in_dim)]
        self.br = [0.0] * n_experts
        self.route_counts = [0] * n_experts

    def num_params(self) -> int:
        per = self.in_dim * self.h + self.h + self.h * self.V + self.V
        router = self.in_dim * self.n_experts + self.n_experts
        return per * self.n_experts + router

    def active_params(self) -> int:
        """Params that actually fire for one token (one expert + router)."""
        per = self.in_dim * self.h + self.h + self.h * self.V + self.V
        return per + self.in_dim * self.n_experts + self.n_experts

    def _active_inputs(self, ctx):
        return [j * self.V + ctx[j] for j in range(self.c)]

    def _route(self, active):
        logits = list(self.br)
        for p in active:
            row = self.Wr[p]
            for e in range(self.n_experts):
                logits[e] += row[e]
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        choice = max(range(self.n_experts), key=lambda e: logits[e])
        return choice, probs

    def _expert_forward(self, exp, active):
        pre1 = list(exp["b1"])
        for p in active:
            row = exp["W1"][p]
            for i in range(self.h):
                pre1[i] += row[i]
        a1 = [math.tanh(x) for x in pre1]
        logits = list(exp["b2"])
        for i in range(self.h):
            ai = a1[i]
            w2 = exp["W2"][i]
            for k in range(self.V):
                logits[k] += ai * w2[k]
        mx = max(logits)
        e = [math.exp(x - mx) for x in logits]
        z = sum(e)
        return a1, [x / z for x in e]

    def forward(self, ctx):
        active = self._active_inputs(ctx)
        choice, _ = self._route(active)
        _, probs = self._expert_forward(self.experts[choice], active)
        return probs

    def nll(self, ctx, target):
        return -math.log(max(self.forward(ctx)[target], 1e-12))

    def train_step(self, ctx, target, lr):
        """One SGD step. Gradient flows only to the chosen expert and its router logit,
        plus a load-balance nudge toward the least-used expert."""
        active = self._active_inputs(ctx)
        choice, rprobs = self._route(active)
        self.route_counts[choice] += 1
        exp = self.experts[choice]
        a1, probs = self._expert_forward(exp, active)
        loss = -math.log(max(probs[target], 1e-12))

        dlogits = list(probs)
        dlogits[target] -= 1.0
        # expert W2/b2
        for i in range(self.h):
            ai = a1[i]
            w2 = exp["W2"][i]
            for k in range(self.V):
                w2[k] -= lr * ai * dlogits[k]
        for k in range(self.V):
            exp["b2"][k] -= lr * dlogits[k]
        # expert hidden
        da1 = [sum(exp["W2"][i][k] * dlogits[k] for k in range(self.V)) for i in range(self.h)]
        dpre = [da1[i] * (1 - a1[i] * a1[i]) for i in range(self.h)]
        for p in active:
            w1 = exp["W1"][p]
            for i in range(self.h):
                w1[i] -= lr * dpre[i]
        for i in range(self.h):
            exp["b1"][i] -= lr * dpre[i]

        # router: push chosen expert's logit toward lower loss, with load balancing.
        # Reward signal: lower loss -> reinforce this route; balance term spreads load.
        total = sum(self.route_counts) or 1
        for e in range(self.n_experts):
            share = self.route_counts[e] / total
            target_share = 1.0 / self.n_experts
            # encourage choosing under-used experts; reinforce chosen if it did well
            advantage = (-loss if e == choice else 0.0)
            balance = -self.balance_coef * (share - target_share)
            grad = -(advantage + balance) * (1.0 if e == choice else 0.0)
            for p in active:
                self.Wr[p][e] -= lr * 0.1 * grad
            self.br[e] -= lr * 0.1 * grad
        return loss

    def load_balance(self):
        """Fraction of tokens to the most-used expert (1/n_experts = perfect balance)."""
        total = sum(self.route_counts) or 1
        return max(self.route_counts) / total


__all__ = ["MoELM"]

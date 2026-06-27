# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A small, readable, rasbt-faithful GPT in PyTorch (optional dependency).

Deliberately close to the canonical decoder-only transformer so every line is
auditable — token + positional embeddings, ``n_layer`` pre-norm blocks (causal
multi-head attention + MLP, residual), final norm, weight-tied LM head. Attention
uses ``F.scaled_dot_product_attention(is_causal=True)`` so it runs efficiently on
CUDA (DGX Spark), MPS (M3), and CPU **without** an x86-only flash-attn wheel.

This is the real model the ``pretraining/`` studies were always analogies for:
the scaling-law fit, optimizer probe, and MoE comparison can now be re-run on an
actual GPT instead of the 1-hidden-layer ``nano`` (see the brainstorm doc).

    from pretraining.gpt.model import GPT, GPTConfig
    m = GPT(GPTConfig(vocab_size=264, block_size=128, n_layer=4, n_head=4, n_embd=128))
    logits, loss = m(idx, targets)        # cross-entropy if targets given
"""
from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover - exercised only without torch
    raise ImportError(
        "pretraining.gpt.model needs PyTorch (optional). Install torch>=2.1 on the "
        "DGX Spark (bf16) or M3 (mps); the tokenizer + research path run without it."
    ) from exc


@dataclass
class GPTConfig:
    vocab_size: int = 264          # 256 bytes + 8 reserved provenance specials
    block_size: int = 128          # context length
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    bias: bool = True
    abstain_head: bool = False     # idea #3: 3-way accept|hedge|abstain head

    def quick(self) -> "GPTConfig":
        """A few-thousand-step CI/smoke config (preserves abstain_head)."""
        return GPTConfig(self.vocab_size, block_size=32, n_layer=2, n_head=2,
                         n_embd=64, dropout=0.0, bias=self.bias,
                         abstain_head=self.abstain_head)


# Decision labels for the optional abstention head (idea #3).
DECISION_LABELS = ("accept", "hedge", "abstain")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying
        # Optional metacognitive head: accept | hedge | abstain (idea #3).
        self.decision_head = nn.Linear(cfg.n_embd, len(DECISION_LABELS)) if cfg.abstain_head else None
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        """Trainable scalars, minus the (tied) position-independent double count."""
        n = sum(p.numel() for p in self.parameters())
        return n - self.transformer.wpe.weight.numel()

    def hidden_states(self, idx):
        """Transformer output (post final-norm), shape (B, T, n_embd)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        return self.transformer.ln_f(x)

    def forward(self, idx, targets=None):
        x = self.hidden_states(idx)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    def decision_logits(self, idx):
        """3-way accept|hedge|abstain logits from the last-token hidden state.
        Raises if the model was built without ``abstain_head=True``."""
        if self.decision_head is None:
            raise RuntimeError("model built without abstain_head=True")
        last = self.hidden_states(idx)[:, -1, :]   # (B, n_embd)
        return self.decision_head(last)            # (B, 3)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k=None):
        """Autoregressive sampling — a sanity check, not a serving path."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def estimate_loss_floor(vocab_size: int) -> float:
    """Uniform-prediction cross-entropy (nats) — the trivial upper bound a trained
    model must beat. Mirrors ``nano``'s known-floor discipline: report fits against
    a reference, never eyeball them."""
    return math.log(vocab_size)


__all__ = ["GPT", "GPTConfig", "estimate_loss_floor"]

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Recurrent-Depth Transformer (RDT) in PyTorch — Phase 1 of the OpenMythos plan.

A from-scratch, re-derived implementation of the looped/recurrent-depth transformer that
``recurrent_depth.py`` validated at nano scale. This is the GPU-trainable version. We do
NOT import OpenMythos's code — its stability ("Parcae") routine is the part most likely to
be subtly wrong and it is the headline, so the LTI injection here is re-derived from first
principles (the S4/Mamba state-space discretization), giving a per-channel diagonal gate
that is provably inside the unit interval:

    A  = -softplus(A_log)        (continuous-time state pole, strictly negative real part)
    dt =  softplus(log_dt)       (strictly positive step)
    a  =  exp(A · dt) ∈ (0, 1)   (zero-order-hold discretization → spectral radius < 1)

so the discrete recurrence ``h_{t+1} = a ⊙ h_t + B·e + block(...)`` has its linear state
term a guaranteed contraction for ANY parameter values — the stability guarantee is
structural, not a training outcome (verified by ``self_test``).

Architecture (matching the OpenMythos prelude→recurrent→coda decomposition):

    embed → prelude blocks (run once)
          → [ LTI-inject(e) → shared recurrent block ] × n_loop   (latent reasoning)
          → coda blocks (run once) → norm → tied LM head

Blocks are standard pre-norm transformer blocks (RMSNorm + GQA attention with RoPE +
SwiGLU MLP, with an optional fine-grained top-k MoE FFN). An optional halting head exposes
the per-loop confidence signal that Phase 2 couples to the Sophia provenance gate
("compute = verification depth").

Honest scope: this is the trainable *architecture*, validated by a CPU self-test (shapes,
finite gradients, the spectral-radius guarantee, and a tiny-batch overfit that proves the
loop learns). It is NOT a trained model or a capability claim. Scaling + the no-overclaim
eval are Phases 1.2–3 in ``RECURRENT-DEPTH.md``.

    python -m pretraining.architecture.rdt_torch --self-test     # CPU, seconds, $0
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RDTConfig:
    vocab_size: int = 256
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int = 4            # GQA: n_heads % n_kv_heads == 0
    d_ff: int = 768               # SwiGLU hidden (dense path)
    n_prelude: int = 2
    n_coda: int = 2
    n_loop: int = 8               # recurrent-block iterations at train time
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    dropout: float = 0.0
    tie_embeddings: bool = True
    use_halt_head: bool = True
    # fine-grained MoE in the recurrent block (DeepSeek-style); dense if False
    use_moe: bool = False
    moe_n_experts: int = 8
    moe_top_k: int = 2
    moe_n_shared: int = 1         # always-on shared experts

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, "n_kv_heads must divide n_heads"
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def _rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                       # [S, hd/2]
    cos = torch.cos(freqs).to(dtype)
    sin = torch.sin(freqs).to(dtype)
    return cos, sin


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, S, hd]; rotate halves
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2] = rx1
    out[..., 1::2] = rx2
    return out


class Attention(nn.Module):
    def __init__(self, cfg: RDTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.d_model // cfg.n_heads
        self.rep = self.n_heads // self.n_kv
        self.wq = nn.Linear(cfg.d_model, self.n_heads * self.hd, bias=False)
        self.wk = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.d_model, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(self.n_heads * self.hd, cfg.d_model, bias=False)
        self.drop = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.n_heads, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        if self.rep > 1:                                   # GQA: expand KV heads
            k = k.repeat_interleave(self.rep, dim=1)
            v = v.repeat_interleave(self.rep, dim=1)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        o = o.transpose(1, 2).contiguous().view(B, S, self.n_heads * self.hd)
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    """Fine-grained top-k MoE with always-on shared experts (DeepSeek-style).

    Returns (output, aux_load_balance_loss). Each expert is a small SwiGLU (d_ff // top_k)
    so active FLOPs stay comparable to a dense FFN. Shared experts run for every token."""
    def __init__(self, cfg: RDTConfig) -> None:
        super().__init__()
        self.n_exp = cfg.moe_n_experts
        self.top_k = cfg.moe_top_k
        self.n_shared = cfg.moe_n_shared
        small_ff = max(1, cfg.d_ff // cfg.moe_top_k)
        self.gate = nn.Linear(cfg.d_model, cfg.moe_n_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(cfg.d_model, small_ff) for _ in range(cfg.moe_n_experts))
        self.shared = nn.ModuleList(SwiGLU(cfg.d_model, small_ff) for _ in range(cfg.moe_n_shared))

    def forward(self, x: torch.Tensor):
        B, S, D = x.shape
        flat = x.reshape(-1, D)                            # [T, D]
        logits = self.gate(flat)                           # [T, E]
        probs = logits.softmax(-1)
        topv, topi = probs.topk(self.top_k, dim=-1)        # [T, k]
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            idx = topi[:, slot]
            w = topv[:, slot].unsqueeze(-1)
            for e in range(self.n_exp):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](flat[mask])
        for sh in self.shared:
            out += sh(flat)
        # Switch-style load-balance aux loss: fraction routed × mean gate prob, ×E.
        with torch.no_grad():
            one_hot = F.one_hot(topi[:, 0], self.n_exp).float()
            frac = one_hot.mean(0)
        aux = self.n_exp * (frac * probs.mean(0)).sum()
        return out.reshape(B, S, D), aux


class Block(nn.Module):
    def __init__(self, cfg: RDTConfig, moe: bool = False) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.is_moe = moe
        self.ffn = MoEFFN(cfg) if moe else SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        if self.is_moe:
            f, aux = self.ffn(self.ffn_norm(x))
            return x + f, aux
        return x + self.ffn(self.ffn_norm(x)), x.new_zeros(())


class LTIInjection(nn.Module):
    """Per-channel diagonal LTI gate + input injection: h ← a ⊙ h + B·e, a ∈ (0,1).

    The discrete pole ``a = exp(-softplus(A_log) · softplus(log_dt))`` is the zero-order-hold
    discretization of a stable continuous-time diagonal system, so 0 < a < 1 for ALL
    parameter values — the spectral radius of the linear state term is < 1 by construction."""
    def __init__(self, d_model: int, min_decay: float = 1e-4) -> None:
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(d_model))
        self.log_dt = nn.Parameter(torch.zeros(d_model))
        self.B = nn.Linear(d_model, d_model, bias=False)
        # A small decay floor so a <= exp(-min_decay) < 1 STRICTLY for every parameter
        # value — without it, a channel whose pole underflows to 0 would give a == 1.0
        # exactly (a lossless integrator: marginally stable, but not a contraction).
        self.min_decay = float(min_decay)

    def a(self) -> torch.Tensor:
        rate = F.softplus(self.A_log) * F.softplus(self.log_dt)
        return torch.exp(-rate - self.min_decay)

    def spectral_radius(self) -> float:
        return float(self.a().max().item())

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.a() * h + self.B(e)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class RDT(nn.Module):
    def __init__(self, cfg: RDTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.prelude = nn.ModuleList(Block(cfg) for _ in range(cfg.n_prelude))
        self.recurrent = Block(cfg, moe=cfg.use_moe)      # ONE shared block, looped
        self.lti = LTIInjection(cfg.d_model)
        self.coda = nn.ModuleList(Block(cfg) for _ in range(cfg.n_coda))
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.halt_head = nn.Linear(cfg.d_model, 1) if cfg.use_halt_head else None
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def spectral_radius(self) -> float:
        return self.lti.spectral_radius()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                n_loop: int | None = None, return_halt: bool = False,
                return_trajectory: bool = False):
        B, S = idx.shape
        cfg = self.cfg
        T = n_loop if n_loop is not None else cfg.n_loop
        cos, sin = _rope_cache(S, cfg.d_model // cfg.n_heads, cfg.rope_theta,
                               idx.device, self.embed.weight.dtype)
        x = self.embed(idx)
        aux_total = x.new_zeros(())
        for blk in self.prelude:
            x, _ = blk(x, cos, sin)
        e = x                                              # injected each loop
        h = e
        halts = []
        traj = []
        for _ in range(T):
            h = self.lti(h, e)                             # a⊙h + B·e
            h, aux = self.recurrent(h, cos, sin)           # + shared transformer block
            aux_total = aux_total + aux
            if self.halt_head is not None:
                halts.append(self.halt_head(self.norm_f(h)).squeeze(-1))
            if return_trajectory:
                # "what would the model say if it halted now" — early-exit readout via the
                # shared head (the per-loop signal VGRD's depth-confidence consumes).
                traj.append(self.lm_head(self.norm_f(h)))
        for blk in self.coda:
            h, _ = blk(h, cos, sin)
        h = self.norm_f(h)
        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size),
                                   targets.reshape(-1), ignore_index=-100)
            if cfg.use_moe:
                loss = loss + 1e-2 * aux_total / max(1, T)
        if return_trajectory:
            return logits, loss, torch.stack(traj, 1)        # [B, T, S, V]
        if return_halt:
            halt = torch.stack(halts, 1) if halts else None  # [B, T, S]
            return logits, loss, halt
        return logits, loss

    @torch.no_grad()
    def loop_trajectory(self, idx: torch.Tensor, pos: int = -1,
                        n_loop: int | None = None) -> "list[list[float]]":
        """Per-loop logit vectors at sequence position ``pos`` for the first batch row, as a
        plain Python list — the exact input ``vgrd.depth_confidence`` / ``vgrd_decide`` expect.
        This is the bridge from the trained RDT to the fail-closed VGRD gate."""
        _, _, traj = self.forward(idx, n_loop=n_loop, return_trajectory=True)
        return [row.tolist() for row in traj[0, :, pos, :]]

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, n_loop: int | None = None,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits, _ = self.forward(idx_cond, n_loop=n_loop)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = logits.softmax(-1)
            nxt = torch.multinomial(probs, 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------------------
# CPU self-test (cheap validation FIRST — the cost-guard contract)
# ---------------------------------------------------------------------------

def self_test(verbose: bool = True) -> dict:
    """Validate the architecture on CPU for $0 before any GPU spend:
    shapes, finite gradients, the spectral-radius guarantee, generate(), and a tiny-batch
    overfit that proves the recurrent loop actually learns. Returns a report dict."""
    torch.manual_seed(0)
    report: dict = {}

    # Two configs: dense and MoE, both small.
    for tag, moe in (("dense", False), ("moe", True)):
        cfg = RDTConfig(vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
                        n_prelude=1, n_coda=1, n_loop=4, max_seq_len=32,
                        use_moe=moe, moe_n_experts=4, moe_top_k=2)
        model = RDT(cfg)
        B, S = 3, 16
        idx = torch.randint(0, cfg.vocab_size, (B, S))
        tgt = torch.randint(0, cfg.vocab_size, (B, S))
        logits, loss, halt = model(idx, tgt, return_halt=True)
        assert logits.shape == (B, S, cfg.vocab_size), logits.shape
        assert torch.isfinite(loss), "loss not finite"
        loss.backward()
        gnorm = math.sqrt(sum(p.grad.pow(2).sum().item()
                              for p in model.parameters() if p.grad is not None))
        assert math.isfinite(gnorm) and gnorm > 0, gnorm
        rho = model.spectral_radius()
        assert rho < 1.0, f"spectral radius {rho} !< 1"
        # spectral radius stays < 1 even after a hostile parameter push
        with torch.no_grad():
            model.lti.A_log.add_(-5.0)   # push pole toward 0 magnitude
            model.lti.log_dt.add_(-5.0)
        assert model.spectral_radius() < 1.0
        report[tag] = {"params": model.num_params(), "loss": round(loss.item(), 4),
                       "grad_norm": round(gnorm, 4), "spectral_radius": round(rho, 6),
                       "halt_shape": list(halt.shape) if halt is not None else None}

    # Depth knob: more loops must change the output (the recurrence is actually used).
    cfg = RDTConfig(vocab_size=64, d_model=64, n_heads=4, n_kv_heads=2, d_ff=128,
                    n_prelude=1, n_coda=1, n_loop=4, max_seq_len=32)
    model = RDT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    with torch.no_grad():
        l2, _ = model(idx, n_loop=2)
        l8, _ = model(idx, n_loop=8)
    depth_changes_output = bool((l2 - l8).abs().max() > 1e-4)
    assert depth_changes_output, "loop count had no effect — recurrence unused"

    # Tiny-batch overfit: the model must drive loss down on a fixed batch (loop learns).
    torch.manual_seed(0)
    model = RDT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, cfg.vocab_size, (4, 16))
    tgt = torch.randint(0, cfg.vocab_size, (4, 16))
    first = last = None
    for step in range(120):
        opt.zero_grad()
        _, loss = model(idx, tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        assert model.spectral_radius() < 1.0  # invariant holds throughout training
    overfit_ok = last < first * 0.5

    # generate() runs and extends the sequence.
    out = model.generate(idx[:1, :4], max_new_tokens=5, top_k=10)
    gen_ok = out.shape == (1, 9)

    report["depth_changes_output"] = depth_changes_output
    report["overfit"] = {"first_loss": round(first, 4), "last_loss": round(last, 4),
                         "reduced_below_half": overfit_ok}
    report["generate_ok"] = gen_ok
    report["all_passed"] = bool(
        depth_changes_output and overfit_ok and gen_ok
        and all(report[t]["spectral_radius"] < 1.0 for t in ("dense", "moe")))
    if verbose:
        import json
        print(json.dumps(report, indent=2))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        rep = self_test(verbose=True)
        raise SystemExit(0 if rep["all_passed"] else 1)
    # default: print a param-count summary for the reference configs
    for name, cfg in [
        ("rdt_nano", RDTConfig(vocab_size=8192, d_model=256, n_heads=8, n_kv_heads=2,
                               d_ff=768, n_prelude=2, n_coda=2, n_loop=8)),
        ("rdt_0p5b", RDTConfig(vocab_size=50304, d_model=1024, n_heads=16, n_kv_heads=4,
                               d_ff=4096, n_prelude=4, n_coda=4, n_loop=8)),
    ]:
        print(f"{name}: {RDT(cfg).num_params()/1e6:.1f}M params  ρ={RDT(cfg).spectral_radius():.4f}")


if __name__ == "__main__":
    main()

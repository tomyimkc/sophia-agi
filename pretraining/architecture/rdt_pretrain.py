#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Pretraining entrypoint for the Recurrent-Depth Transformer (Phase 1).

One script that runs a tiny CPU **smoke** locally ($0, the cost-guard "cheap validation
FIRST" rule) and the same code path scales to a GPU pretrain. It deliberately mirrors the
OpenMythos 3B recipe knobs (AdamW, linear-warmup→cosine, bf16, FineWeb-Edu) so the GPU run
is a config change, not a rewrite.

Data: streams ``HuggingFaceFW/fineweb-edu`` when ``datasets`` + network are available;
otherwise falls back to a deterministic synthetic byte-stream so the smoke is hermetic and
runnable anywhere (including this no-network CI box). Tokenizer: a byte-level tokenizer by
default (vocab 256, zero deps) so the smoke needs nothing gated; pass ``--hf-tokenizer`` to
use a real BPE on GPU.

    # CPU smoke — hermetic, seconds, proves the training loop + checkpoint round-trip
    python -m pretraining.architecture.rdt_pretrain --smoke

    # GPU pretrain (on a pod), single process
    python -m pretraining.architecture.rdt_pretrain --steps 20000 --batch 32 --seq 1024 \
        --d-model 1024 --n-loop 8 --bf16 --dataset fineweb-edu --out ckpt/rdt-0p5b

Honest scope: this validates the *pipeline* (data→loss→optimizer→checkpoint) and the
architecture's trainability. It is NOT a trained model or a capability claim; the scaling
run + the no-overclaim eval are the next steps in ``RECURRENT-DEPTH.md``.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from pretraining.architecture.rdt_torch import RDT, RDTConfig


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _synthetic_stream(vocab: int, n_tokens: int, seed: int = 0) -> torch.Tensor:
    """Deterministic order-2 byte stream with learnable structure (hermetic fallback)."""
    g = torch.Generator().manual_seed(seed)
    # a peaky transition table over `vocab` so loss can fall below uniform (log vocab).
    table = torch.softmax(torch.randn(vocab, vocab, generator=g) * 3.0, dim=-1)
    out = torch.empty(n_tokens, dtype=torch.long)
    out[0] = 0
    prev = 0
    u = torch.rand(n_tokens, generator=g)
    cdf = table.cumsum(-1)
    for i in range(1, n_tokens):
        out[i] = int(torch.searchsorted(cdf[prev], u[i]))
        prev = int(out[i])
    return out


def _fineweb_stream(vocab: int, n_tokens: int):
    """Stream FineWeb-Edu bytes if datasets+network available; else None."""
    try:
        from datasets import load_dataset
    except Exception:
        return None
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
        buf = bytearray()
        for row in ds:
            buf.extend(row["text"].encode("utf-8", "ignore"))
            buf.append(0)
            if len(buf) >= n_tokens:
                break
        return torch.tensor(list(buf[:n_tokens]), dtype=torch.long).clamp_(0, vocab - 1)
    except Exception:
        return None


def _batches(stream: torch.Tensor, batch: int, seq: int, device, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    n = stream.numel()
    while True:
        ix = torch.randint(0, n - seq - 1, (batch,), generator=g)
        x = torch.stack([stream[i:i + seq] for i in ix]).to(device)
        y = torch.stack([stream[i + 1:i + 1 + seq] for i in ix]).to(device)
        yield x, y


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(args) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = RDTConfig(
        vocab_size=args.vocab, d_model=args.d_model, n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads, d_ff=args.d_ff, n_prelude=args.n_prelude,
        n_coda=args.n_coda, n_loop=args.n_loop, max_seq_len=args.seq, use_moe=args.moe)
    model = RDT(cfg).to(device)
    dtype = torch.bfloat16 if (args.bf16 and device.type == "cuda") else torch.float32

    # data
    stream = None
    src = "synthetic"
    if args.dataset == "fineweb-edu":
        stream = _fineweb_stream(args.vocab, args.data_tokens)
        src = "fineweb-edu" if stream is not None else "synthetic(fallback)"
    if stream is None:
        stream = _synthetic_stream(args.vocab, args.data_tokens, seed=args.seed)
    gen = _batches(stream, args.batch, args.seq, device, seed=args.seed)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * prog))

    log: list[dict] = []
    t0 = time.time()
    model.train()
    for step in range(args.steps):
        for pg in opt.param_groups:
            pg["lr"] = lr_at(step)
        x, y = next(gen)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            _, loss, _ = model(x, y)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        assert model.spectral_radius() < 1.0  # the stability invariant holds every step
        if step % args.log_every == 0 or step == args.steps - 1:
            rec = {"step": step, "loss": round(loss.item(), 4),
                   "lr": round(lr_at(step), 6), "grad_norm": round(float(gnorm), 3),
                   "spectral_radius": round(model.spectral_radius(), 5)}
            log.append(rec)
            print(json.dumps(rec))

    elapsed = time.time() - t0
    first, last = log[0]["loss"], log[-1]["loss"]
    report = {
        "data_source": src, "device": str(device), "dtype": str(dtype),
        "params": model.num_params(), "config": cfg.__dict__,
        "steps": args.steps, "elapsed_s": round(elapsed, 2),
        "first_loss": first, "last_loss": last,
        "loss_decreased": last < first, "uniform_loss": round(math.log(args.vocab), 4),
        "below_uniform": last < math.log(args.vocab),
        "final_spectral_radius": round(model.spectral_radius(), 5),
        "log": log,
    }

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": cfg.__dict__}, out / "ckpt.pt")
        (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n")
        # checkpoint round-trip sanity (proves the artifact is loadable)
        blob = torch.load(out / "ckpt.pt", map_location="cpu", weights_only=False)
        m2 = RDT(RDTConfig(**blob["config"]))
        m2.load_state_dict(blob["model"])
        report["checkpoint_roundtrip_ok"] = True
        (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _apply_smoke(args) -> None:
    args.steps = 60
    args.warmup = 5
    args.batch = 8
    args.seq = 32
    args.vocab = 64
    args.d_model = 64
    args.n_heads = 4
    args.n_kv_heads = 2
    args.d_ff = 128
    args.n_prelude = 1
    args.n_coda = 1
    args.n_loop = 4
    args.data_tokens = 20000
    args.device = "cpu"
    args.bf16 = False
    args.log_every = 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny hermetic CPU run ($0)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--d-model", dest="d_model", type=int, default=1024)
    ap.add_argument("--n-heads", dest="n_heads", type=int, default=16)
    ap.add_argument("--n-kv-heads", dest="n_kv_heads", type=int, default=4)
    ap.add_argument("--d-ff", dest="d_ff", type=int, default=4096)
    ap.add_argument("--n-prelude", dest="n_prelude", type=int, default=4)
    ap.add_argument("--n-coda", dest="n_coda", type=int, default=4)
    ap.add_argument("--n-loop", dest="n_loop", type=int, default=8)
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", dest="min_lr", type=float, default=6e-5)
    ap.add_argument("--grad-clip", dest="grad_clip", type=float, default=1.0)
    ap.add_argument("--dataset", choices=["synthetic", "fineweb-edu"], default="synthetic")
    ap.add_argument("--data-tokens", dest="data_tokens", type=int, default=2_000_000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--log-every", dest="log_every", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        _apply_smoke(args)
        if args.out is None:
            args.out = "/tmp/rdt-smoke"
    rep = train(args)
    print(json.dumps({k: v for k, v in rep.items() if k != "log"}, indent=2))
    if args.smoke:
        ok = rep["loss_decreased"] and rep["below_uniform"] and rep.get("checkpoint_roundtrip_ok")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Train the from-scratch GPT — device-agnostic across the dev cluster.

Runs on CPU (CI), the DGX Spark (CUDA/bf16), or the M3 (MPS) with no code change:
the tier is resolved by ``pretraining.gpt.cluster.resolve_tier``. Logs the same
``epoch_loss`` / ``grad_norms`` shape as ``pretraining/nano`` and the optimizer
probe, so the existing analysis tooling reads its output directly. Writes a dated
``gpt-train-latest.json`` report stamped ``canClaimAGI: false``.

    python -m pretraining.gpt.train --quick            # seconds, CI/smoke
    python -m pretraining.gpt.train --steps 2000       # a real laptop/Spark run
    python -m pretraining.gpt.train --prefer cuda --steps 5000 --report
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from pretraining.gpt.cluster import resolve_tier, torch_dtype
from pretraining.gpt.data import token_stream, train_val_split
from pretraining.gpt.model import GPT, GPTConfig, estimate_loss_floor
from pretraining.gpt.tokenizer import ByteProvenanceTokenizer

HERE = Path(__file__).resolve().parent


def _get_batch(data, block_size, batch_size, device, rng):
    import torch  # noqa: PLC0415

    ix = [rng.randint(0, len(data) - block_size - 1) for _ in range(batch_size)]
    x = torch.tensor([data[i:i + block_size] for i in ix], dtype=torch.long)
    y = torch.tensor([data[i + 1:i + 1 + block_size] for i in ix], dtype=torch.long)
    return x.to(device), y.to(device)


def _grad_global_norm(model) -> float:
    import torch  # noqa: PLC0415

    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum())
    return total ** 0.5


def train(*, quick: bool = False, steps: int = 2000, batch_size: int = 16,
          lr: float = 3e-4, prefer: str = "auto", seed: int = 0,
          eval_every: int = 0) -> dict:
    """Real cross-entropy descent. Returns a JSON-able report dict."""
    import torch  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = random.Random(seed)

    tier = resolve_tier(prefer)
    device = tier.device

    tok = ByteProvenanceTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size)
    if quick:
        cfg = cfg.quick()
        steps = min(steps, 60)

    ids = token_stream(tok)
    train_ids, val_ids = train_val_split(ids)
    if len(train_ids) <= cfg.block_size + 1:  # tiny synthetic corpus → repeat
        train_ids = (train_ids * (cfg.block_size * 4 // max(1, len(train_ids)) + 2))
        val_ids = train_ids[-(cfg.block_size + 2):]

    model = GPT(cfg).to(device)
    # bf16/fp16 autocast on accelerators; plain fp32 on CPU.
    use_amp = device in {"cuda", "mps"}
    amp_dtype = torch_dtype(tier) if use_amp else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    epoch_loss: list[float] = []
    grad_norms: list[float] = []
    model.train()
    for step in range(steps):
        x, y = _get_batch(train_ids, cfg.block_size, batch_size, device, rng)
        if use_amp:
            with torch.autocast(device_type=device, dtype=amp_dtype):
                _, loss = model(x, y)
        else:
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = _grad_global_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        epoch_loss.append(float(loss.detach()))
        grad_norms.append(gnorm)

    # Held-out loss (one batch is enough for a smoke signal).
    model.eval()
    with torch.no_grad():
        vx, vy = _get_batch(val_ids if len(val_ids) > cfg.block_size + 1 else train_ids,
                            cfg.block_size, batch_size, device, rng)
        _, vloss = model(vx, vy)

    floor = estimate_loss_floor(cfg.vocab_size)
    report = {
        "canClaimAGI": False,
        "boundary": "from-scratch GPT smoke/iteration run — not headline evidence; "
                    "headline numbers stay on x86 RunPod (see docs/11-Platform/DGX-Spark.md)",
        "tier": tier.as_dict(),
        "config": cfg.__dict__,
        "num_params": model.num_params(),
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "uniform_loss_floor_nats": round(floor, 4),
        "first_loss": round(epoch_loss[0], 4) if epoch_loss else None,
        "final_loss": round(epoch_loss[-1], 4) if epoch_loss else None,
        "val_loss": round(float(vloss), 4),
        "beats_uniform": bool(float(vloss) < floor),
        "epoch_loss": [round(x, 4) for x in epoch_loss],
        "grad_norms": [round(x, 4) for x in grad_norms],
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the from-scratch Sophia GPT.")
    ap.add_argument("--quick", action="store_true", help="tiny config, ~60 steps (CI/smoke)")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--prefer", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true", help="write gpt-train-latest.json")
    args = ap.parse_args(argv)

    try:
        report = train(quick=args.quick, steps=args.steps, batch_size=args.batch_size,
                       lr=args.lr, prefer=args.prefer, seed=args.seed)
    except ImportError as exc:
        print(f"[gpt.train] {exc}")
        return 2

    summary = {k: report[k] for k in
               ("tier", "num_params", "first_loss", "final_loss", "val_loss",
                "uniform_loss_floor_nats", "beats_uniform")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.report:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (HERE / "gpt-train-latest.json").write_text(
            json.dumps({**report, "generatedAt": stamp}, indent=2, ensure_ascii=False),
            encoding="utf-8")
        print(f"[gpt.train] wrote {HERE / 'gpt-train-latest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

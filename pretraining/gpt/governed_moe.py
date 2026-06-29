# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Governed MoE (idea #7): adopt sparse routing ONLY with its trust governor.

Per [Governed-Scaling.md](../../docs/11-Platform/Governed-Scaling.md): an
efficiency primitive is admitted only when it carries its own proof. For MoE that
means two governors, both fail-closed (default = reject):

  - **Governor #4 — equivalence / error bound.** The MoE variant is accepted only
    if its loss is within a pre-registered relative bound of the dense baseline at
    matched active compute (top_k=1). "Faster/bigger" is never enough.
  - **Governor #3 — monoculture alarm.** Routing must not collapse onto a few
    experts. We watch normalised routing entropy; below a floor the model is
    leaning on a monoculture of experts and is rejected even if loss looks fine.

The decision logic is dependency-free and CI-tested; the run that produces the
measured numbers is torch-gated. ``canClaimAGI: false``.

    python -m pretraining.gpt.governed_moe --quick
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def normalised_entropy(load: "list[float]") -> float:
    """Routing entropy / log(n_experts) in [0, 1]. 1.0 = perfectly balanced,
    →0 = collapsed onto one expert (monoculture)."""
    n = len(load)
    if n <= 1:
        return 1.0
    total = sum(load) or 1.0
    p = [x / total for x in load]
    h = -sum(pi * math.log(pi) for pi in p if pi > 0)
    return h / math.log(n)


def governed_moe_decision(
    dense_loss: float, moe_loss: float, expert_loads: "list[list[float]]",
    *, rel_bound: float = 0.05, entropy_floor: float = 0.5,
) -> dict:
    """Fail-closed verdict. Accept MoE only if loss is within ``rel_bound`` of
    dense AND no block's routing entropy falls below ``entropy_floor``."""
    rel_gap = (moe_loss - dense_loss) / max(abs(dense_loss), 1e-9)
    within_bound = rel_gap <= rel_bound          # MoE may match OR beat dense
    entropies = [normalised_entropy(b) for b in expert_loads] or [1.0]
    min_entropy = min(entropies)
    no_collapse = min_entropy >= entropy_floor

    accept = bool(within_bound and no_collapse)
    if accept:
        verdict, reason = "accept", "within error bound and no routing collapse"
    elif not within_bound:
        verdict, reason = "reject", f"loss gap {rel_gap:.3f} exceeds bound {rel_bound}"
    else:
        verdict, reason = "reject", f"routing collapse: entropy {min_entropy:.3f} < {entropy_floor}"

    return {
        "canClaimAGI": False,
        "verdict": verdict,                      # accept | reject (fail-closed)
        "reason": reason,
        "dense_loss": round(dense_loss, 4),
        "moe_loss": round(moe_loss, 4),
        "rel_gap": round(rel_gap, 4),
        "rel_bound": rel_bound,
        "within_bound": within_bound,
        "min_routing_entropy": round(min_entropy, 4),
        "entropy_floor": entropy_floor,
        "no_collapse": no_collapse,
    }


def run_governed_moe(*, quick: bool = False, experts: int = 4, steps: int = 600,
                     seed: int = 0, rel_bound: float = 0.05) -> dict:
    """Train dense vs top-1 MoE at matched active compute; apply the governor."""
    import torch  # noqa: PLC0415

    from pretraining.gpt.model import GPT, GPTConfig, expert_load  # noqa: PLC0415
    from pretraining.gpt.train import train  # noqa: PLC0415

    if quick:
        steps, experts = 50, 3

    dense = train(quick=quick, steps=steps, prefer="cpu", seed=seed)

    # MoE arm at top_k=1 (matched active compute). Re-run the loop with MoE config
    # by temporarily building the model through train()'s machinery would need a
    # config hook; instead train a MoE model inline and read its val loss + load.
    import random as _r

    from pretraining.gpt.data import token_stream, train_val_split  # noqa: PLC0415
    from pretraining.gpt.tokenizer import ByteProvenanceTokenizer  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = _r.Random(seed)
    tok = ByteProvenanceTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size, moe_experts=experts, moe_top_k=1)
    if quick:
        cfg = cfg.quick()
    ids = token_stream(tok)
    tr, va = train_val_split(ids)
    if len(tr) <= cfg.block_size + 2:
        tr = tr * (cfg.block_size * 4 // max(1, len(tr)) + 2)
        va = tr[-(cfg.block_size + 2):]

    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    model.train()
    bs = 16
    for _ in range(steps):
        ix = [rng.randint(0, len(tr) - cfg.block_size - 1) for _ in range(bs)]
        xb = torch.tensor([tr[i:i + cfg.block_size] for i in ix], dtype=torch.long)
        yb = torch.tensor([tr[i + 1:i + 1 + cfg.block_size] for i in ix], dtype=torch.long)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        ix = [rng.randint(0, len(va) - cfg.block_size - 1) for _ in range(bs)] \
            if len(va) > cfg.block_size + 1 else [0] * bs
        src = va if len(va) > cfg.block_size + 1 else tr
        xb = torch.tensor([src[i:i + cfg.block_size] for i in ix], dtype=torch.long)
        yb = torch.tensor([src[i + 1:i + 1 + cfg.block_size] for i in ix], dtype=torch.long)
        _, vloss = model(xb, yb)
        loads = expert_load(model)

    decision = governed_moe_decision(dense["val_loss"], float(vloss), loads,
                                     rel_bound=rel_bound)
    decision["experts"] = experts
    decision["expert_loads"] = loads
    return decision


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Governed MoE vs dense (trust governor).")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--rel-bound", type=float, default=0.05)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    try:
        rep = run_governed_moe(quick=args.quick, experts=args.experts,
                               steps=args.steps, rel_bound=args.rel_bound)
    except ImportError as exc:
        print(f"[gpt.governed_moe] {exc}")
        return 2
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.report:
        (HERE / "gpt-governed-moe-latest.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

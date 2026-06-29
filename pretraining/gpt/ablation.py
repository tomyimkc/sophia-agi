# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Born-gated vs plain-text ablation — the experiment behind idea #1.

The falsifiable question: *does training with inline provenance markers
(`<src>`/`<conf>`/`<doNotAttributeTo>`) make a from-scratch GPT assert fewer
forbidden attributions than the same architecture trained on plain text?*

Procedure (pre-registered, fail-closed):
  1. train two identical-config GPTs — one on the plain corpus, one born-gated;
  2. prompt both with attribution questions ("Who wrote the {title}?");
  3. score continuations with the dependency-free
     ``provenance_eval.forbidden_attribution_rate`` (lower is better);
  4. report the delta and whether born-gated wins.

Honest boundary: at nano scale on a tiny corpus both models are weak, so a *null*
or noisy result is expected and legitimate — the deliverable is the **measurable
ablation**, stamped ``canClaimAGI: false``, not a headline. Run multiple seeds on
the cluster before reading anything into the sign.

    python -m pretraining.gpt.ablation --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretraining.gpt.born_gated import load_records
from pretraining.gpt.provenance_eval import forbidden_attribution_rate
from pretraining.gpt.tokenizer import ByteProvenanceTokenizer

HERE = Path(__file__).resolve().parent


def attribution_prompts(limit: int = 16) -> "list[str]":
    """Held-out probes: ask who wrote each titled work."""
    prompts: list[str] = []
    for r in load_records():
        title = r.get("canonicalTitleEn") or r.get("canonicalTitle")
        if title:
            prompts.append(f"Who wrote the {title}? ")
        if len(prompts) >= limit:
            break
    return prompts or ["Who wrote the Dao De Jing? "]


def _sample(model, tok, prompt, max_new_tokens, device):
    import torch  # noqa: PLC0415

    ids = tok.encode(prompt)[-model.cfg.block_size:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    return tok.decode(out[0].tolist())


def run_ablation(*, quick: bool = False, steps: int = 800, seed: int = 0,
                 max_new_tokens: int = 48) -> dict:
    import torch  # noqa: PLC0415

    # Train both arms inline (we need the live model to sample from, not just
    # metrics), reusing the same config + optimiser settings as train.py.
    from pretraining.gpt.born_gated import born_gated_token_stream  # noqa: PLC0415
    from pretraining.gpt.data import token_stream  # noqa: PLC0415
    from pretraining.gpt.model import GPT, GPTConfig  # noqa: PLC0415

    tok = ByteProvenanceTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size)
    if quick:
        cfg = cfg.quick()
        steps = min(steps, 60)
        max_new_tokens = min(max_new_tokens, 16)

    device = "cpu"
    prompts = attribution_prompts(4 if quick else 16)

    def _train_arm(ids, arm_seed):
        torch.manual_seed(arm_seed)
        import random as _r
        rng = _r.Random(arm_seed)
        m = GPT(cfg).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4, betas=(0.9, 0.95))
        m.train()
        bs = 16
        for _ in range(steps):
            ix = [rng.randint(0, max(1, len(ids) - cfg.block_size - 1)) for _ in range(bs)]
            xb = torch.tensor([ids[i:i + cfg.block_size] for i in ix], dtype=torch.long, device=device)
            yb = torch.tensor([ids[i + 1:i + 1 + cfg.block_size] for i in ix], dtype=torch.long, device=device)
            _, loss = m(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        m.eval()
        return m

    plain_ids = token_stream(tok)
    bg_ids = born_gated_token_stream(tok)
    # pad tiny born-gated stream so batching is valid
    while len(bg_ids) <= cfg.block_size + 2:
        bg_ids = bg_ids * 2

    plain_model = _train_arm(plain_ids, seed)
    bg_model = _train_arm(bg_ids, seed + 1)

    plain_out = [_sample(plain_model, tok, p, max_new_tokens, device) for p in prompts]
    bg_out = [_sample(bg_model, tok, p, max_new_tokens, device) for p in prompts]

    plain_rate = forbidden_attribution_rate(plain_out)
    bg_rate = forbidden_attribution_rate(bg_out)

    return {
        "canClaimAGI": False,
        "boundary": "nano-scale born-gated ablation — illustrative; a null/noisy "
                    "result is expected at this scale. Multi-seed on the cluster "
                    "before interpreting the sign.",
        "n_prompts": len(prompts),
        "steps": steps,
        "plain_forbidden_rate": round(plain_rate, 4),
        "born_gated_forbidden_rate": round(bg_rate, 4),
        "delta_plain_minus_bg": round(plain_rate - bg_rate, 4),
        "born_gated_better": bg_rate < plain_rate,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Born-gated vs plain-text provenance ablation.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    try:
        rep = run_ablation(quick=args.quick, steps=args.steps, seed=args.seed)
    except ImportError as exc:
        print(f"[gpt.ablation] {exc}")
        return 2
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.report:
        (HERE / "gpt-ablation-latest.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

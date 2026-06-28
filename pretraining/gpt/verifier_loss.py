# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-in-the-loss (idea #2): make the gate a training signal, not just a filter.

`provenance_eval.provenance_penalty` turns "would `agent/gate.py` block this?" into
a scalar. This module uses it two ways:

  - **DPO-style preference loss** — prefer the gate-passing ``chosen`` over the
    lineage-merging ``rejected`` (the cheap, stable entry point: no reward model,
    no RL, just a margin on sequence log-probs).
  - **reward hook** — ``sequence_reward(text)`` = ``-penalty``, ready for the
    existing RLVR stack (`provenance_bench/rl_reward.py`, `tools/run_rlvr.py`) if
    you later want token-level RL.

Dependency-free pieces (penalty, pair construction) are CI-tested; the DPO step is
torch-gated and stamps ``canClaimAGI: false``. Honest boundary: a margin on a tiny
model is a *mechanism demo*, not a capability result.

    python -m pretraining.gpt.verifier_loss --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pretraining.gpt.provenance_eval import preference_pairs, provenance_penalty
from pretraining.gpt.tokenizer import ByteProvenanceTokenizer

HERE = Path(__file__).resolve().parent


def sequence_reward(text: str) -> float:
    """Verifier reward: 0 if faithful, -1 if it merges a forbidden lineage."""
    return -provenance_penalty(text)


def verified_pairs() -> "list[dict]":
    """Preference pairs whose labels the verifier actually agrees with — chosen
    must score 0 penalty, rejected must score 1. Fail-closed: drop any pair the
    verifier doesn't confirm, so the training signal is never self-contradictory."""
    out = []
    for p in preference_pairs():
        if provenance_penalty(p["chosen"]) == 0.0 and provenance_penalty(p["rejected"]) == 1.0:
            out.append(p)
    return out


def _seq_logprob(model, tok, text, block_size, device):
    """Mean per-token log-prob of ``text`` under the model (teacher forcing)."""
    import torch  # noqa: PLC0415

    ids = tok.encode(text)[: block_size + 1]
    if len(ids) < 2:
        ids = ids + [tok.eot_id]
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    logp = torch.log_softmax(logits, dim=-1)[0]
    tok_lp = logp[range(y.shape[1]), y[0]]
    return tok_lp.mean()


def run_dpo(*, quick: bool = False, steps: int = 200, beta: float = 0.1,
            lr: float = 1e-3, seed: int = 0) -> dict:
    import torch  # noqa: PLC0415

    from pretraining.gpt.model import GPT, GPTConfig  # noqa: PLC0415

    torch.manual_seed(seed)
    tok = ByteProvenanceTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size)
    if quick:
        cfg = cfg.quick()
        steps = min(steps, 40)

    pairs = verified_pairs()
    if not pairs:
        return {"canClaimAGI": False, "error": "no verifier-confirmed pairs"}

    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    first = last = None
    for step in range(steps):
        total = 0.0
        opt.zero_grad(set_to_none=True)
        for p in pairs:
            lp_c = _seq_logprob(model, tok, p["chosen"], cfg.block_size, "cpu")
            lp_r = _seq_logprob(model, tok, p["rejected"], cfg.block_size, "cpu")
            # DPO loss vs a uniform reference (ref log-probs cancel to 0 here):
            loss = -torch.nn.functional.logsigmoid(beta * (lp_c - lp_r))
            loss.backward()
            total += float(loss)
        opt.step()
        if step == 0:
            first = total / len(pairs)
        last = total / len(pairs)

    return {
        "canClaimAGI": False,
        "boundary": "verifier-DPO mechanism demo on a tiny model — not a capability "
                    "result; margin shows the gate signal is trainable, nothing more.",
        "n_pairs": len(pairs),
        "steps": steps,
        "beta": beta,
        "first_loss": round(first, 4) if first is not None else None,
        "final_loss": round(last, 4) if last is not None else None,
        "loss_decreased": (last is not None and first is not None and last < first),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verifier-in-the-loss DPO (provenance reward).")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    try:
        rep = run_dpo(quick=args.quick, steps=args.steps)
    except ImportError as exc:
        print(f"[gpt.verifier_loss] {exc}")
        return 2
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.report:
        (HERE / "gpt-verifier-dpo-latest.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

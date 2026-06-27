# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Train + evaluate the abstention head (idea #3): accept | hedge | abstain.

VISION.md marks the self-model / calibration pillar ⚠️ *partial* because
confidence today is a threshold rule, not a learned signal. This trains the GPT's
optional 3-way ``decision_head`` so the model itself predicts whether a
continuation should be **accepted, hedged, or abstained** — and scores it with the
repo's own calibration tooling (`agent/calibration.py`: ECE, risk-coverage).

The supervision is the provenance signal, not hand labels: a faithful (correct,
source-marked) statement → ``accept``; a lineage merge → ``abstain``. So the head
learns to flag exactly what the gate would block.

Torch-gated; stamps ``canClaimAGI: false``. A tiny-model run is *illustrative* —
the deliverable is the wired, measurable mechanism.

    python -m pretraining.gpt.abstain --quick
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pretraining.gpt.provenance_eval import preference_pairs
from pretraining.gpt.tokenizer import ByteProvenanceTokenizer

HERE = Path(__file__).resolve().parent

ACCEPT, HEDGE, ABSTAIN = 0, 1, 2


def decision_dataset() -> "list[tuple[str, int]]":
    """(text, label) pairs from the provenance preference set. ``chosen`` →
    accept; ``rejected`` (a forbidden merge) → abstain."""
    data: list[tuple[str, int]] = []
    for p in preference_pairs():
        data.append((p["chosen"], ACCEPT))
        data.append((p["rejected"], ABSTAIN))
    return data


def _encode_block(tok: ByteProvenanceTokenizer, text: str, block_size: int) -> "list[int]":
    """Encode and LEFT-pad/truncate to block_size so the real last token stays
    last (the decision head reads the final position)."""
    ids = tok.encode(text)[-block_size:]
    if len(ids) < block_size:
        ids = [tok.eot_id] * (block_size - len(ids)) + ids
    return ids


def train_and_eval(*, quick: bool = False, steps: int = 300, lr: float = 1e-3,
                   seed: int = 0) -> dict:
    import torch  # noqa: PLC0415

    from agent.calibration import calibration_report  # noqa: PLC0415
    from pretraining.gpt.model import DECISION_LABELS, GPT, GPTConfig  # noqa: PLC0415

    torch.manual_seed(seed)
    rng = random.Random(seed)

    tok = ByteProvenanceTokenizer()
    cfg = GPTConfig(vocab_size=tok.vocab_size, abstain_head=True)
    if quick:
        cfg = cfg.quick()
        steps = min(steps, 60)

    data = decision_dataset()
    rng.shuffle(data)
    split = max(1, int(len(data) * 0.8))
    train_set, val_set = data[:split], data[split:] or data[:2]

    X = torch.tensor([_encode_block(tok, t, cfg.block_size) for t, _ in train_set], dtype=torch.long)
    Y = torch.tensor([y for _, y in train_set], dtype=torch.long)

    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        logits = model.decision_logits(X)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Evaluate on the held-out split: accuracy + calibration of P(accept-ish).
    model.eval()
    confidences: list[float] = []
    correct: list[bool] = []
    n_right = 0
    with torch.no_grad():
        for text, label in val_set:
            xb = torch.tensor([_encode_block(tok, text, cfg.block_size)], dtype=torch.long)
            probs = torch.softmax(model.decision_logits(xb), dim=-1)[0]
            pred = int(torch.argmax(probs))
            n_right += int(pred == label)
            # "confidence" = how strongly the head commits to its top decision
            confidences.append(float(probs.max()))
            correct.append(pred == label)

    return {
        "canClaimAGI": False,
        "boundary": "abstention-head mechanism wired + measured on a tiny model — "
                    "illustrative, not a calibration claim (single tiny GPT, small N).",
        "labels": list(DECISION_LABELS),
        "n_train": len(train_set),
        "n_val": len(val_set),
        "val_accuracy": round(n_right / len(val_set), 4),
        "calibration": calibration_report(confidences, correct),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train+eval the GPT abstention head.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    try:
        rep = train_and_eval(quick=args.quick, steps=args.steps)
    except ImportError as exc:
        print(f"[gpt.abstain] {exc}")
        return 2
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.report:
        (HERE / "gpt-abstain-latest.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""ORPO discipline trainer — sharpen abstention/citation via preference pairs.

Abstention and source-citation are *contrastive* behaviours ("this answer fabricated"
vs "this answer abstained/cited"), which preference methods instil better than plain
SFT. This trains a LoRA adapter with ORPO (Odds-Ratio Preference Optimization) — a
single-stage, reference-model-free method (one trainer, no separate SFT/DPO reference)
— on the gate-disciplined preference pairs the repo already produces.

Pairs come from tools/wiki_to_training.py (chosen = lineage-correct + provenance-aware,
rejected = forbidden-attribution merge). Run that first:

    python tools/wiki_to_training.py
    python tools/train_orpo.py --dry-run
    python tools/train_orpo.py --4bit --rslora --epochs 1

Recommended pattern: SFT (tools/train_lora.py) to fix FORMAT, then ORPO here to sharpen
the discipline contrast. Mix in "unanswerable/uncitable" negatives to avoid the
refusal-forgetting "hallucination tax". CUDA-only (needs trl/peft/bitsandbytes);
install: pip install -r requirements-rl.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PAIRS = ROOT / "training" / "wiki_provenance_dpo.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def load_pairs(path: Path) -> list[dict]:
    """Load {prompt, chosen, rejected} preference rows; skip malformed/degenerate ones."""
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("prompt") and r.get("chosen") and r.get("rejected") and r["chosen"] != r["rejected"]:
            rows.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    ap.add_argument("--output", type=Path, default=ROOT / "training" / "lora" / "checkpoints" / "sophia-orpo-v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6, help="ORPO uses a lower LR than SFT")
    ap.add_argument("--beta", type=float, default=0.1, help="ORPO lambda: odds-ratio loss weight")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-prompt-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--rslora", dest="use_rslora", action="store_true")
    ap.add_argument("--target-modules", choices=("attn-mlp", "all-linear"), default="all-linear")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--4bit", dest="four_bit", action="store_true")
    ap.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.pairs.exists():
        print(f"Missing {args.pairs}. Run: python tools/wiki_to_training.py", flush=True)
        return 1
    pairs = load_pairs(args.pairs)
    print(f"Preference pairs: {len(pairs)} | model: {args.model} | output: {args.output}", flush=True)
    if pairs:
        print(f"  sample prompt: {pairs[0]['prompt'][:70]}", flush=True)
    if args.dry_run or not pairs:
        return 0

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from trl import ORPOConfig, ORPOTrainer
    except Exception as exc:  # noqa: BLE001
        print(f"Install RL deps: pip install -r requirements-rl.txt ({type(exc).__name__}: {exc})", flush=True)
        traceback.print_exc(file=sys.stdout)
        return 1

    if not torch.cuda.is_available():
        print("CUDA GPU not detected; ORPO training is CUDA-only.", flush=True)
        return 1

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    dtype = torch.bfloat16 if (args.dtype == "bf16" or (args.dtype == "auto" and torch.cuda.is_bf16_supported())) else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if args.four_bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    else:
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)

    target = "all-linear" if args.target_modules == "all-linear" else \
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM", use_rslora=args.use_rslora, target_modules=target)

    ds = Dataset.from_list(pairs)
    cfg = ORPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_len,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        logging_steps=10,
        save_strategy="epoch",
        seed=args.seed,
        report_to=[],
    )
    trainer = ORPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tokenizer, peft_config=peft_config)
    trainer.train()
    trainer.save_model(str(args.output))
    (args.output / "sophia_orpo_config.json").write_text(json.dumps({
        "baseModel": args.model, "method": "orpo", "pairs": len(pairs), "beta": args.beta,
        "lr": args.lr, "epochs": args.epochs, "loraR": args.lora_r, "loraAlpha": args.lora_alpha,
        "rslora": args.use_rslora, "targetModules": args.target_modules, "seed": args.seed,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved ORPO adapter to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""DPO discipline trainer — hard-negative preference pairs on top of an SFT LoRA.

Stage 3 of sophia-7b-train-verify: sharpen abstention / citation via TRL DPOTrainer
on gate-validated {prompt, chosen, rejected} rows (e.g. training/local_sophia_7b/dpo_hard_negatives.jsonl).

Requires a prior SFT adapter (--adapter). CUDA-only; install: pip install -r requirements-rl.txt

    python tools/train_dpo.py --dry-run
    python tools/train_dpo.py --4bit --rslora --adapter training/lora/checkpoints/sophia-cuda-v1 \\
        --pairs training/local_sophia_7b/dpo_hard_negatives.jsonl --epochs 1
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

DEFAULT_PAIRS = ROOT / "training" / "local_sophia_7b" / "dpo_hard_negatives.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"


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
    ap.add_argument("--adapter", type=Path, required=False,
                    help="existing SFT LoRA directory to continue training")
    ap.add_argument("--output", type=Path, default=ROOT / "training" / "lora" / "checkpoints" / "sophia-dpo-v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty weight)")
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
        print(f"Missing {args.pairs}.", flush=True)
        return 1
    pairs = load_pairs(args.pairs)
    print(f"DPO pairs: {len(pairs)} | model: {args.model} | adapter: {args.adapter} | output: {args.output}",
          flush=True)
    if pairs:
        print(f"  sample prompt: {pairs[0]['prompt'][:70]}", flush=True)
    if args.dry_run or not pairs:
        return 0
    if not args.adapter or not args.adapter.exists():
        print("Missing --adapter (SFT LoRA directory). DPO runs on top of Stage-2 SFT.", flush=True)
        return 1

    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from trl import DPOConfig, DPOTrainer
    except Exception as exc:  # noqa: BLE001
        print(f"Install RL deps: pip install -r requirements-rl.txt ({type(exc).__name__}: {exc})", flush=True)
        traceback.print_exc(file=sys.stdout)
        return 1

    if not torch.cuda.is_available():
        print("CUDA GPU not detected; DPO training is CUDA-only.", flush=True)
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
    base = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=True)

    ds = Dataset.from_list(pairs)
    cfg = DPOConfig(
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
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(args.output))
    (args.output / "sophia_dpo_config.json").write_text(json.dumps({
        "baseModel": args.model,
        "method": "dpo",
        "sftAdapter": str(args.adapter),
        "pairs": len(pairs),
        "pairsPath": str(args.pairs),
        "beta": args.beta,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "label": "release gate — NOT third-party evidence",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved DPO adapter to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)

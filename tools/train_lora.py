#!/usr/bin/env python3
"""Minimal LoRA SFT for Sophia AGI corpus (optional GPU workflow).

Install: pip install -r requirements-lora.txt
Prepare: python tools/prepare_lora_dataset.py

Usage:
  python tools/train_lora.py --dry-run
  python tools/train_lora.py --epochs 3 --output training/lora/checkpoints/sophia-v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = ROOT / "training" / "lora" / "train.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune on Sophia corpus")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=TRAIN_JSONL)
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "lora" / "checkpoints" / "sophia-v1")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="QLoRA 4-bit load")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.train.exists():
        print(f"Missing {args.train}. Run: python tools/prepare_lora_dataset.py")
        return 1

    rows = load_rows(args.train)
    print(f"Train rows: {len(rows)} | model: {args.model} | output: {args.output}")
    if args.dry_run or not rows:
        return 0

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import SFTTrainer
    except ImportError:
        print("Install LoRA deps: pip install -r requirements-lora.txt")
        return 1

    if not torch.cuda.is_available():
        print("CUDA GPU not detected. Use Google Colab: notebooks/Sophia-LoRA-Colab.ipynb")
        return 1

    if args.four_bit:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
        except ImportError:
            print("4-bit requires bitsandbytes. Colab: pip install bitsandbytes")
            return 1

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"trust_remote_code": True, "device_map": "auto"}
    if args.four_bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = Dataset.from_list([{"text": row["text"]} for row in rows])
    args.output.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        processing_class=tokenizer,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
    )
    trainer.train()
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "sophia_lora_config.json").write_text(
        json.dumps({
            "baseModel": args.model,
            "trainRows": len(rows),
            "epochs": args.epochs,
            "loraR": args.lora_r,
            "loraAlpha": args.lora_alpha,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved adapter to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Minimal LoRA SFT for Sophia AGI corpus (GPU / Colab workflow).

Uses a minimal manual SFT loop + PEFT (no HF Trainer — avoids Windows Trainer import crash).

Install: pip install -r requirements-lora.txt
Prepare: python tools/prepare_lora_dataset.py

Usage:
  python tools/train_lora.py --dry-run
  python tools/train_lora.py --4bit --epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = ROOT / "training" / "lora" / "train.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
DONE_MARKER = ".train_complete"


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_checkpoint(output: Path) -> list[str]:
    return [name for name in REQUIRED_ADAPTER_FILES if not (output / name).exists()]


def build_model_and_tokenizer(
    model_id: str,
    four_bit: bool,
    lora_r: int,
    lora_alpha: int,
    *,
    resume_adapter: Path | None = None,
) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if four_bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if four_bit:
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    if resume_adapter and resume_adapter.exists():
        model = PeftModel.from_pretrained(model, str(resume_adapter), is_trainable=True)
        print(f"Resumed adapter from {resume_adapter}")
    else:
        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def tokenize_dataset(tokenizer: Any, rows: list[dict], max_seq_len: int) -> Any:
    from datasets import Dataset

    ds = Dataset.from_list([{"text": row["text"]} for row in rows])

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )

    return ds.map(tokenize_batch, batched=True, remove_columns=["text"])


def run_manual_train(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_accum: int,
    four_bit: bool,
) -> None:
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    device = model.get_input_embeddings().weight.device
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if four_bit:
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam8bit(trainable, lr=lr)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr)

    model.train()
    total_steps = max(1, (len(loader) * epochs) // grad_accum)
    global_step = 0

    print(f"Manual SFT: {epochs} epoch(s), batch={batch_size}, grad_accum={grad_accum}", flush=True)
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % 10 == 0 or global_step == 1:
                    pct = round(100.0 * global_step / total_steps, 1)
                    print(f"epoch {epoch + 1}/{epochs} step {global_step}/{total_steps} ({pct}%) loss={loss.item() * grad_accum:.4f}", flush=True)

        if len(loader) % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def save_adapter(model: Any, tokenizer: Any, output: Path, base_model: str, meta: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    payload = {"baseModel": base_model, **meta}
    (output / "sophia_lora_config.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / DONE_MARKER).write_text("ok\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune on Sophia corpus")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=TRAIN_JSONL)
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "lora" / "checkpoints" / "sophia-v1")
    parser.add_argument(
        "--resume-adapter",
        type=Path,
        default=None,
        help="Continue training from an existing PEFT adapter (e.g. sophia-v1 -> sophia-v2)",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="QLoRA 4-bit load")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.train.exists():
        print(f"Missing {args.train}. Run: python tools/prepare_lora_dataset.py")
        return 1

    rows = load_rows(args.train)
    print(f"Train rows: {len(rows)} | model: {args.model} | output: {args.output}", flush=True)
    if args.dry_run or not rows:
        return 0

    print("loading torch...", flush=True)
    try:
        import torch
        print("loading transformers...", flush=True)
        from transformers import AutoModelForCausalLM  # noqa: F401 — warmup import
    except Exception as exc:
        print(f"Install LoRA deps: pip install -r requirements-lora.txt ({type(exc).__name__}: {exc})", flush=True)
        traceback.print_exc(file=sys.stdout)
        return 1

    print(f"cuda available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("CUDA GPU not detected. Use Google Colab: notebooks/Sophia-LoRA-Colab.ipynb", flush=True)
        return 1

    if args.four_bit:
        print("loading bitsandbytes...", flush=True)
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            print(f"4-bit requires bitsandbytes. Colab: pip install bitsandbytes ({exc})", flush=True)
            return 1

    print(f"torch={torch.__version__}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / DONE_MARKER
    if marker.exists():
        marker.unlink()

    model, tokenizer = build_model_and_tokenizer(
        args.model,
        args.four_bit,
        args.lora_r,
        args.lora_alpha,
        resume_adapter=args.resume_adapter,
    )
    train_ds = tokenize_dataset(tokenizer, rows, args.max_seq_len)

    run_manual_train(
        model,
        tokenizer,
        train_ds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum=4,
        four_bit=args.four_bit,
    )

    meta = {
        "trainRows": len(rows),
        "epochs": args.epochs,
        "loraR": args.lora_r,
        "loraAlpha": args.lora_alpha,
    }
    save_adapter(model, tokenizer, args.output, args.model, meta)

    missing = verify_checkpoint(args.output)
    if missing:
        print(f"ERROR: Training finished but adapter files missing: {missing}")
        return 1

    print(f"Saved adapter to {args.output}")
    print("Checkpoint files:", sorted(p.name for p in args.output.iterdir()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)
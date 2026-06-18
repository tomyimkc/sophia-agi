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
import inspect
import json
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN_JSONL = ROOT / "training" / "lora" / "train.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def verify_checkpoint(output: Path) -> list[str]:
    return [name for name in REQUIRED_ADAPTER_FILES if not (output / name).exists()]


def filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = set(inspect.signature(cls.__init__).parameters)
    return {key: value for key, value in kwargs.items() if key in params}


def save_checkpoint(trainer: Any, output: Path, base_model: str, meta: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    if hasattr(trainer, "save_model"):
        trainer.save_model(str(output))
    else:
        trainer.model.save_pretrained(output)

    tokenizer = getattr(trainer, "processing_class", None) or getattr(trainer, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.save_pretrained(output)

    meta_payload = {"baseModel": base_model, **meta}
    (output / "sophia_lora_config.json").write_text(
        json.dumps(meta_payload, indent=2) + "\n",
        encoding="utf-8",
    )


def build_modern_trainer(
    model_id: str,
    train_dataset: Any,
    *,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_seq_len: int,
    four_bit: bool,
    lora_r: int,
    lora_alpha: int,
) -> Any | None:
    """TRL >= 1.0: SFTConfig + peft_config (recommended on Colab)."""
    import torch
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    model_init_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if four_bit:
        from transformers import BitsAndBytesConfig

        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_init_kwargs["torch_dtype"] = torch.float16

    config_kwargs = filter_kwargs(
        SFTConfig,
        {
            "output_dir": str(output_dir),
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": 4,
            "learning_rate": lr,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "report_to": "none",
            "gradient_checkpointing": True,
            "fp16": True,
            "bf16": False,
            "max_length": max_seq_len,
            "max_seq_length": max_seq_len,
            "model_init_kwargs": model_init_kwargs,
            "dataset_text_field": "text",
        },
    )
    training_args = SFTConfig(**config_kwargs)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    trainer_kwargs = filter_kwargs(
        SFTTrainer,
        {
            "model": model_id,
            "args": training_args,
            "train_dataset": train_dataset,
            "peft_config": peft_config,
        },
    )
    print("Using TRL modern trainer (SFTConfig + peft_config)")
    return SFTTrainer(**trainer_kwargs)


def build_legacy_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    *,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    max_seq_len: int,
) -> Any:
    """TRL 0.9.x fallback with pre-wrapped PeftModel."""
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer

    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer_params = set(trainer_sig.parameters)

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
    }
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer_kwargs["args"] = training_args
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = max_seq_len
    elif "max_length" in trainer_params:
        trainer_kwargs["max_length"] = max_seq_len

    print("Using TRL legacy trainer (pre-wrapped PeftModel)")
    try:
        return SFTTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("dataset_text_field", None)
        trainer_kwargs.pop("max_seq_length", None)
        trainer_kwargs.pop("max_length", None)
        return SFTTrainer(**trainer_kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune on Sophia corpus")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train", type=Path, default=TRAIN_JSONL)
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "lora" / "checkpoints" / "sophia-v1")
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
    print(f"Train rows: {len(rows)} | model: {args.model} | output: {args.output}")
    if args.dry_run or not rows:
        return 0

    try:
        import torch
        from datasets import Dataset
    except ImportError:
        print("Install LoRA deps: pip install -r requirements-lora.txt")
        return 1

    if not torch.cuda.is_available():
        print("CUDA GPU not detected. Use Google Colab: notebooks/Sophia-LoRA-Colab.ipynb")
        return 1

    if args.four_bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            print("4-bit requires bitsandbytes. Colab: pip install bitsandbytes")
            return 1

    try:
        import trl

        print(f"torch={torch.__version__} trl={trl.__version__}")
    except Exception:
        pass

    ds = Dataset.from_list([{"text": row["text"]} for row in rows])
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {
        "trainRows": len(rows),
        "epochs": args.epochs,
        "loraR": args.lora_r,
        "loraAlpha": args.lora_alpha,
    }

    try:
        trainer = build_modern_trainer(
            args.model,
            ds,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_seq_len=args.max_seq_len,
            four_bit=args.four_bit,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
        )
    except (ImportError, TypeError, ValueError) as exc:
        print(f"Modern TRL path failed ({exc}); falling back to legacy path")
        trainer = None

    if trainer is None:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError:
            print("Install LoRA deps: pip install -r requirements-lora.txt")
            return 1

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
        if args.four_bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        if args.four_bit:
            model = prepare_model_for_kbit_training(model)
        model.gradient_checkpointing_enable()

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
        trainer = build_legacy_trainer(
            model,
            tokenizer,
            ds,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_seq_len=args.max_seq_len,
        )

    trainer.train()
    save_checkpoint(trainer, args.output, args.model, meta)

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
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Minimal LoRA SFT for Sophia AGI corpus (GPU / Colab workflow).

Uses a minimal manual SFT loop + PEFT (no HF Trainer — avoids Windows Trainer import crash).

Design notes (why this is more than a toy loop):
  * Dynamic padding (pad-to-longest-in-batch), NOT pad-to-max-length — short rows no
    longer pay a full ``max_seq_len`` forward/backward. This is the single biggest
    wall-clock win on a short, curated corpus.
  * Completion-only loss (``--mask-prompt``, default on) — loss is computed on the
    assistant turn only, matching the MLX-LM ``--mask-prompt`` path so the two
    backends are comparable.
  * Holdout eval loop + early stopping (``--eval-every`` / ``--patience``) — the
    eval ladder is wired INTO training, so "stop before you overfit" is automatic
    rather than eyeballed afterwards.
  * Provenance-discipline data hooks: ``--scaffold`` (inject the advisor system
    prompt where missing), ``--guard`` (drop any training target that trips the
    deterministic gate), ``--distill`` (fold in gate-clean council traces).
  * Reproducibility/stability: ``--seed`` (emitted into the adapter config so the
    promotion gate can verify it), cosine LR + warmup, gradient clipping, bf16 by
    default on Ada/Hopper.

Install: pip install -r requirements-lora.txt
Prepare: python tools/prepare_lora_dataset.py

Usage:
  python tools/train_lora.py --dry-run
  python tools/train_lora.py --4bit --epochs 1 --guard --scaffold --distill
  python tools/train_lora.py --4bit --eval-every 25 --patience 4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_JSONL = ROOT / "training" / "lora" / "train.jsonl"
HOLDOUT_JSONL = ROOT / "training" / "lora" / "holdout.jsonl"
DISTILL_JSONL = ROOT / "training" / "council" / "traces.jsonl"
MATH_CODE_PACK_DIR = ROOT / "training" / "sophia-math-code-curriculum"
MATH_CODE_SFT = MATH_CODE_PACK_DIR / "sft_all.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
REQUIRED_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
DONE_MARKER = ".train_complete"


# --------------------------------------------------------------------------- #
# Data loading + provenance-discipline hooks
# --------------------------------------------------------------------------- #
def load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def resolve_train_path(path: Path) -> Path:
    """Resolve a training JSONL path or pack directory to a concrete JSONL file.

    When ``path`` is a directory containing ``manifest.json`` with schema
    ``sophia.math_code_curriculum.v1``, returns ``sft_all.jsonl`` inside that pack.
    """
    if path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema") == "sophia.math_code_curriculum.v1":
                return path / "sft_all.jsonl"
    return path


def scaffold_rows(rows: list[dict]) -> int:
    """Ensure every row carries the advisor source-discipline system prompt.

    Rows that already have a system turn are left untouched. Returns the count of
    rows that received an injected scaffold.
    """
    from agent.prompts import MODE_PROMPTS

    system_prompt = MODE_PROMPTS["advisor"]
    injected = 0
    for row in rows:
        msgs = row.get("messages") or []
        if not any(m.get("role") == "system" for m in msgs):
            row["messages"] = [{"role": "system", "content": system_prompt}, *msgs]
            injected += 1
    return injected


def guard_filter(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop any row whose assistant target carries an INTRINSIC gate violation.

    Fail-closed at the data layer for the things that are wrong regardless of the
    prompt: a fabricated/nonexistent legal citation, false arithmetic, or a
    forbidden-lineage attribution merge embedded in the answer text. This is the
    safety net for distilled/synthetic targets (``--distill``) that were not
    hand-curated.

    Deliberately does NOT pass the question. With a question, ``check_response``
    also runs the attribution *trap grader* ("expected discussion of socrates",
    "expected tradition context 'daoist'"), which is a positive-expectation
    completeness check — a clean gold answer phrased differently fails it. Using
    that here would silently delete ~16% of the hand-curated corpus over wording,
    not fabrication. Intrinsic-only checking flags 0/439 curated rows (verified)
    while still catching genuine fabrication in synthetic targets.

    Returns (kept, dropped).
    """
    from agent.gate import check_response

    kept: list[dict] = []
    dropped = 0
    for row in rows:
        msgs = row.get("messages") or []
        target = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "assistant"), "")
        if not target:
            dropped += 1
            continue
        violations = check_response(target, mode="advisor")["violations"]
        if violations:
            dropped += 1
        else:
            kept.append(row)
    return kept, dropped


def load_distill_rows(path: Path) -> list[dict]:
    """Fold in gate-clean council-distillation traces as extra SFT targets."""
    if not path.exists():
        print(f"--distill: {path} not found; skipping", flush=True)
        return []
    rows = load_rows(path)
    print(f"--distill: folding in {len(rows)} council trace(s) from {path}", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Tokenization: prompt/completion split for completion-only loss
# --------------------------------------------------------------------------- #
def split_prompt_completion(messages: list[dict]) -> tuple[str, str]:
    """Reconstruct the prepare_lora_dataset chat format, split at the assistant turn.

    Prompt = everything up to and including the ``<|assistant|>`` header.
    Completion = the assistant content plus the ``<|end|>`` terminator.
    """
    last_asst = max(i for i, m in enumerate(messages) if m.get("role") == "assistant")
    prefix_parts: list[str] = []
    for m in messages[:last_asst]:
        content = str(m.get("content", "")).strip()
        if content:
            prefix_parts.append(f"<|{m.get('role', 'user')}|>\n{content}")
    prompt = "\n".join(prefix_parts) + "\n<|assistant|>\n"
    completion = str(messages[last_asst].get("content", "")).strip() + "\n<|end|>"
    return prompt, completion


def build_records(
    tokenizer: Any, rows: list[dict], max_seq_len: int, *, mask_prompt: bool
) -> tuple[list[dict], int]:
    """Tokenize rows into variable-length {input_ids, labels} records.

    No padding here — the collator pads per batch. With ``mask_prompt`` the prompt
    tokens are set to -100 so loss lands only on the assistant turn. Returns
    (records, truncated_count); a non-zero truncated_count is surfaced loudly so
    over-long rows are never dropped silently.
    """
    records: list[dict] = []
    truncated = 0
    for row in rows:
        msgs = row.get("messages")
        if msgs and any(m.get("role") == "assistant" for m in msgs):
            prompt, completion = split_prompt_completion(msgs)
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        else:
            full_ids = tokenizer(row.get("text", ""), add_special_tokens=False)["input_ids"]
            prompt_ids = []

        if len(full_ids) > max_seq_len:
            truncated += 1
            full_ids = full_ids[:max_seq_len]

        labels = list(full_ids)
        if mask_prompt and prompt_ids:
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
        records.append({"input_ids": full_ids, "labels": labels})
    return records, truncated


class DynamicCausalCollator:
    """Pad a batch to the longest sequence in that batch (not to max_seq_len).

    ``pad_to`` forces a fixed pad length instead (used by ``--pad-to-max`` to
    reproduce the old pad-to-max-length behaviour for the speedup ablation).
    """

    def __init__(self, pad_token_id: int, pad_to: int | None = None) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to = pad_to

    def __call__(self, features: list[dict]) -> dict:
        import torch

        max_len = self.pad_to or max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids = f["input_ids"]
            lab = f["labels"]
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def _resolve_dtype(choice: str) -> Any:
    import torch

    if choice == "bf16":
        return torch.bfloat16
    if choice == "fp16":
        return torch.float16
    # auto: prefer bf16 on hardware that supports it (Ada/Hopper), else fp16.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


ATTN_MLP_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _resolve_target_modules(choice: str) -> Any:
    # "all-linear" lets PEFT target every linear layer (the strongest module finding:
    # MLP is the dominant locus of adaptation); "attn-mlp" is the explicit Qwen list.
    return "all-linear" if choice == "all-linear" else ATTN_MLP_MODULES


def build_model_and_tokenizer(
    model_id: str,
    four_bit: bool,
    lora_r: int,
    lora_alpha: int,
    *,
    dtype: Any,
    lora_dropout: float = 0.05,
    use_rslora: bool = False,
    target_modules: str = "attn-mlp",
    attn_impl: str | None = None,
    resume_adapter: Path | None = None,
) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if attn_impl:
        load_kwargs["attn_implementation"] = attn_impl  # e.g. flash_attention_2 / sdpa
    if four_bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        load_kwargs["torch_dtype"] = dtype

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
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=use_rslora,
            target_modules=_resolve_target_modules(target_modules),
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model, tokenizer


def attach_neftune(model: Any, alpha: float) -> Any:
    """NEFTune: add uniform noise to embedding outputs during training (regularizer
    that helps instruction tuning, most on small data). Returns the hook handle so the
    caller can remove it before saving/inference. No-op when alpha <= 0."""
    import torch

    if alpha <= 0:
        return None
    emb = model.get_input_embeddings()

    def hook(module: Any, args: Any, output: Any) -> Any:
        if not module.training:
            return output
        dims = torch.tensor(output.size(1) * output.size(2), dtype=torch.float32)
        mag = alpha / torch.sqrt(dims)
        return output + torch.zeros_like(output).uniform_(-mag.item(), mag.item())

    return emb.register_forward_hook(hook)


def build_model_and_tokenizer_unsloth(
    model_id: str,
    four_bit: bool,
    lora_r: int,
    lora_alpha: int,
    *,
    max_seq_len: int,
    dtype: Any,
    seed: int,
    use_rslora: bool = False,
    target_modules: str = "attn-mlp",
    resume_adapter: Path | None = None,
) -> tuple[Any, Any]:
    """Unsloth fused-kernel backend (~2× throughput / ~½ memory vs vanilla PEFT).

    Unsloth handles 4-bit loading and k-bit prep internally; the returned model is a
    standard PEFT model, so the manual training loop below is unchanged.
    """
    from unsloth import FastLanguageModel  # lazy: pip install unsloth (CUDA-only)

    if resume_adapter and resume_adapter.exists():
        raise SystemExit(
            "--backend unsloth does not support --resume-adapter; use --backend peft to "
            "continue from an existing adapter."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_len,
        dtype=dtype,
        load_in_4bit=four_bit,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,  # Unsloth-optimized path requires dropout 0
        bias="none",
        target_modules=_resolve_target_modules(target_modules),
        use_rslora=use_rslora,
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )
    model.print_trainable_parameters()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# MLX backend (Apple Silicon) — invoke mlx_lm with logic kept in-repo
# --------------------------------------------------------------------------- #
def run_mlx_backend(args: argparse.Namespace, rows: list[dict]) -> int:
    """Train via mlx-lm on Apple Silicon. Builds the MLX chat-data dir from the
    (already scaffolded/distilled/guarded) rows, fits every row under the token
    budget, then invokes ``python -m mlx_lm lora``. Emits the seed into the adapter
    config so the promotion gate can verify it."""
    import math
    import subprocess

    from tools.split_long_training_rows import fit_rows

    # Honest limitations of the MLX path (mlx_lm owns the inner loop):
    ignored = [n for n, v in (("--pack", args.pack), ("--rslora", args.use_rslora),
                              ("--neftune-alpha", args.neftune_alpha), ("--weight-decay", args.weight_decay),
                              ("--lora-dropout", args.lora_dropout != 0.05)) if v]
    if ignored:
        print(f"NOTE: {', '.join(ignored)} are peft/unsloth-only and ignored on --backend mlx "
              f"(mlx_lm owns the inner loop). Use --backend peft to apply them.", flush=True)
    if args.pad_to_max:
        print("NOTE: --pad-to-max is a no-op on --backend mlx (mlx_lm controls padding). "
              "Run the padding ablation on --backend peft (CUDA).", flush=True)
    if args.eval_every and args.holdout.exists():
        print("NOTE: Sophia early-stopping (--patience/--overfit-ratio) is peft/unsloth-only. "
              "On mlx, mlx_lm reports validation every --steps-per-eval but does NOT early-stop; "
              "it runs all iters. Inspect its Val loss output to choose --iters.", flush=True)

    def _to_mlx(rs: list[dict]) -> list[dict]:
        return [{"messages": r["messages"], "metadata": r.get("metadata", {})}
                for r in rs if r.get("messages")]

    data_dir = args.output.with_name(args.output.name + "-mlx-data")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_fitted, train_rep = fit_rows(_to_mlx(rows), max_tokens=args.max_seq_len)
    (data_dir / "train.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_fitted), encoding="utf-8")

    valid_n = 0
    if args.holdout.exists():
        hold = load_rows(args.holdout)
        if args.scaffold:
            scaffold_rows(hold)
        valid_fitted, _ = fit_rows(_to_mlx(hold), max_tokens=args.max_seq_len)
        valid_n = len(valid_fitted)
        (data_dir / "valid.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in valid_fitted), encoding="utf-8")

    iters = args.iters or args.epochs * max(1, math.ceil(len(train_fitted) / args.batch_size))
    print(f"MLX data: {len(train_fitted)} train / {valid_n} valid rows → {data_dir} "
          f"(fit: {json.dumps(train_rep)})", flush=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora", "--train",
        "--model", args.model,
        "--data", str(data_dir),
        "--iters", str(iters),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.lr),
        "--adapter-path", str(args.output),
        "--max-seq-length", str(args.max_seq_len),
        "--seed", str(args.seed),
        "--steps-per-report", "50",
        "--save-every", "250",
    ]
    if args.mask_prompt:
        cmd.append("--mask-prompt")
    if valid_n:
        cmd += ["--steps-per-eval", str(args.eval_every or 250)]
    if args.resume_adapter:
        cmd += ["--resume-adapter-file", str(args.resume_adapter / "adapters.safetensors")]

    args.output.mkdir(parents=True, exist_ok=True)
    print("MLX command:\n  " + " ".join(cmd), flush=True)
    try:
        import mlx_lm  # noqa: F401 — fail fast with a clear message
    except Exception as exc:  # noqa: BLE001
        print(f"--backend mlx requires mlx-lm: pip install mlx-lm ({type(exc).__name__}: {exc})", flush=True)
        return 1

    rc = subprocess.run(cmd, cwd=ROOT).returncode

    # Emit the Sophia config (seed + provenance flags) alongside the MLX adapter so
    # the promotion gate sees the same metadata the PEFT path writes.
    (args.output / "sophia_lora_config.json").write_text(
        json.dumps({
            "baseModel": args.model, "backend": "mlx-lm", "seed": args.seed,
            "iters": iters, "batchSize": args.batch_size, "maskPrompt": args.mask_prompt,
            "trainRows": len(train_fitted), "maxSeqLength": args.max_seq_len,
            "scaffold": args.scaffold, "guard": args.guard, "distill": args.distill,
        }, indent=2) + "\n", encoding="utf-8")

    if rc != 0:
        print(f"mlx_lm exited {rc}", flush=True)
        return rc
    if not (args.output / "adapters.safetensors").exists():
        print(f"ERROR: MLX run finished but {args.output / 'adapters.safetensors'} missing", flush=True)
        return 1
    (args.output / DONE_MARKER).write_text("ok\n", encoding="utf-8")
    print(f"Saved MLX adapter to {args.output}", flush=True)
    # Consistent machine-parseable summary (loss lines come from mlx_lm's own stdout above;
    # mlx_lm runs all iters, so earlyStopped is always False on this backend).
    print("Run summary: " + json.dumps({
        "backend": "mlx-lm", "globalSteps": iters, "trainRows": len(train_fitted),
        "validRows": valid_n, "seed": args.seed, "earlyStopped": False,
        "note": "train/val loss + peak mem are in mlx_lm stdout above; no Sophia early-stop on mlx",
    }), flush=True)
    return 0


# --------------------------------------------------------------------------- #
# Train + eval
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001 - numpy optional at train time
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: Any, loader: Any, device: Any, max_batches: int) -> float:
    import torch

    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            total += model(**batch).loss.item()
            n += 1
    model.train()
    return total / max(1, n)


def run_manual_train(
    model: Any,
    tokenizer: Any,
    train_records: list[dict],
    eval_records: list[dict] | None,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    grad_accum: int,
    four_bit: bool,
    warmup_ratio: float,
    max_grad_norm: float,
    eval_every: int,
    eval_batches: int,
    patience: int,
    min_delta: float,
    overfit_ratio: float,
    save_best: "callable | None",
    pad_to: int | None = None,
    weight_decay: float = 0.0,
    pack: bool = False,
) -> dict:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    device = model.get_input_embeddings().weight.device
    # Packing concatenates short rows into one flat sequence (no padding) and relies on
    # Flash-Attention varlen via position_ids to keep examples from attending across
    # boundaries. Completion-only -100 label masks are preserved by the collator.
    if pack:
        from transformers import DataCollatorWithFlattening

        train_collator = DataCollatorWithFlattening()
    else:
        train_collator = DynamicCausalCollator(tokenizer.pad_token_id, pad_to=pad_to)
    # Eval always uses padded batches so val-loss is defined identically across configs.
    eval_collator = DynamicCausalCollator(tokenizer.pad_token_id, pad_to=pad_to)
    loader = DataLoader(train_records, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
    eval_loader = None
    if eval_records:
        eval_loader = DataLoader(eval_records, batch_size=batch_size, shuffle=False, collate_fn=eval_collator)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if four_bit:
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam8bit(trainable, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    steps_per_epoch = max(1, -(-len(loader) // grad_accum))  # ceil
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    best_val = float("inf")
    bad_evals = 0
    saved_any = False
    stop = False
    last_train_loss = float("nan")

    print(
        f"Manual SFT: {epochs} epoch(s), batch={batch_size}, grad_accum={grad_accum}, "
        f"total_opt_steps≈{total_steps}, warmup={warmup_steps}, "
        f"eval_every={eval_every if eval_loader else 'off'}",
        flush=True,
    )
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / grad_accum
            loss.backward()
            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                last_train_loss = loss.item() * grad_accum
                if global_step % 10 == 0 or global_step == 1:
                    pct = round(100.0 * global_step / total_steps, 1)
                    print(
                        f"epoch {epoch + 1}/{epochs} step {global_step}/{total_steps} "
                        f"({pct}%) loss={last_train_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                        flush=True,
                    )

                if eval_loader and eval_every and global_step % eval_every == 0:
                    val = evaluate(model, eval_loader, device, eval_batches)
                    ratio = val / last_train_loss if last_train_loss else float("nan")
                    print(
                        f"  [eval] step {global_step} val_loss={val:.4f} "
                        f"train_loss={last_train_loss:.4f} val/train={ratio:.2f}",
                        flush=True,
                    )
                    if val < best_val - min_delta:
                        best_val = val
                        bad_evals = 0
                        if save_best:
                            save_best({"bestValLoss": round(best_val, 4), "atStep": global_step})
                            saved_any = True
                            print(f"  [eval] new best val_loss — checkpoint saved at step {global_step}", flush=True)
                    else:
                        bad_evals += 1
                        print(f"  [eval] no improvement ({bad_evals}/{patience})", flush=True)
                    if overfit_ratio and ratio > overfit_ratio:
                        print(f"  [eval] STOP: val/train {ratio:.2f} > overfit ratio {overfit_ratio}", flush=True)
                        stop = True
                    if bad_evals >= patience:
                        print(f"  [eval] EARLY STOP: no improvement for {patience} evals", flush=True)
                        stop = True
                if stop:
                    break

        # flush a trailing partial accumulation window
        if len(loader) % grad_accum != 0 and not stop:
            torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if stop:
            break

    return {
        "globalSteps": global_step,
        "finalTrainLoss": round(last_train_loss, 4) if last_train_loss == last_train_loss else None,
        "bestValLoss": round(best_val, 4) if best_val != float("inf") else None,
        "earlyStopped": stop,
        "savedBest": saved_any,
    }


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


def verify_checkpoint(output: Path) -> list[str]:
    return [name for name in REQUIRED_ADAPTER_FILES if not (output / name).exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA fine-tune on Sophia corpus")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--train", "--data", type=Path, default=TRAIN_JSONL, dest="train",
        help="Training JSONL or math-code curriculum pack directory "
             f"(default: {TRAIN_JSONL.relative_to(ROOT)}; pack dir → sft_all.jsonl)",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "training" / "lora" / "checkpoints" / "sophia-v1")
    parser.add_argument(
        "--resume-adapter",
        type=Path,
        default=None,
        help="Continue training from an existing PEFT adapter (e.g. sophia-v1 -> sophia-v2)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="LoRA LR transfer rule (2026 'LoRA Without Regret'): optimal LoRA LR is "
                             "~rank-independent and ~10x the full-FT LR, so a per-rank sweep is usually "
                             "unnecessary — pick by base-model/full-FT LR, not by --lora-r. Default 5e-5 "
                             "(lowered from 2e-4).")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Small-data regularizer; 0.01–0.05 is cheap insurance")
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--rslora", dest="use_rslora", action="store_true",
                        help="Rank-stabilized LoRA: scale by alpha/sqrt(r) (fixes over-aggressive alpha/r)")
    parser.add_argument("--target-modules", choices=("attn-mlp", "all-linear"), default="all-linear",
                        help="all-linear (default) targets every linear layer — the 2026 'LoRA Without "
                             "Regret' finding: matching full-FT by adapting all linear layers (MLP is the "
                             "dominant locus of adaptation); 'attn-mlp' is the narrower explicit Qwen list")
    parser.add_argument("--neftune-alpha", type=float, default=0.0,
                        help="NEFTune embedding-noise regularizer (try 5); helps instruction tuning on small data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16"), default="auto")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="QLoRA 4-bit load")
    parser.add_argument("--attn", choices=("auto", "flash_attention_2", "sdpa", "eager"), default="auto",
                        help="Attention impl; flash_attention_2 is required for --pack")
    parser.add_argument("--pack", action="store_true",
                        help="Sequence packing (concat short rows, no padding) via Flash-Attention varlen")
    parser.add_argument("--backend", choices=("peft", "unsloth", "mlx"), default="peft",
                        help="peft = vanilla PEFT (CUDA); unsloth = fused kernels (CUDA); mlx = Apple Silicon")
    parser.add_argument("--iters", type=int, default=0,
                        help="MLX backend: optimizer steps (0 = derive from epochs × rows/batch)")
    parser.add_argument("--pad-to-max", action="store_true",
                        help="Ablation: pad every batch to --max-seq-len (reproduces pre-fix behaviour)")
    # provenance-discipline data hooks
    parser.add_argument("--scaffold", action="store_true",
                        help="Inject the advisor source-discipline system prompt into rows that lack one")
    parser.add_argument("--guard", action="store_true",
                        help="Drop any training target that trips the deterministic provenance gate")
    parser.add_argument("--distill", action="store_true",
                        help="Fold in gate-clean council-distillation traces as extra SFT targets")
    parser.add_argument("--distill-file", type=Path, default=DISTILL_JSONL)
    parser.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false",
                        help="Train on prompt tokens too (default: completion-only loss)")
    # eval ladder / early stopping
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_JSONL)
    parser.add_argument("--eval-every", type=int, default=0,
                        help="Run holdout eval every N optimizer steps (0 = auto: 25 if holdout exists)")
    parser.add_argument("--eval-batches", type=int, default=0,
                        help="Cap eval batches per round (0 = full holdout)")
    parser.add_argument("--patience", type=int, default=4,
                        help="Stop after this many eval rounds without val-loss improvement")
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--overfit-ratio", type=float, default=0.0,
                        help="Stop if val/train loss exceeds this (0 = disabled)")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(mask_prompt=True)
    args = parser.parse_args()

    # 'LoRA Without Regret' (2026): LoRA on small, curated data prefers an effective
    # batch < 32. Warn (don't fail) when batch_size*grad_accum exceeds that — large
    # effective batches wash out the per-example signal a tiny corpus depends on.
    eff_batch = args.batch_size * args.grad_accum
    if eff_batch > 32:
        print(
            f"WARNING: effective batch size = batch_size({args.batch_size}) * "
            f"grad_accum({args.grad_accum}) = {eff_batch} > 32. For LoRA on small, "
            f"curated data the 2026 'LoRA Without Regret' finding advises an effective "
            f"batch < 32; consider lowering --batch-size or --grad-accum.",
            flush=True,
        )

    args.train = resolve_train_path(args.train)
    if not args.train.exists():
        hint = "python tools/prepare_lora_dataset.py"
        if MATH_CODE_PACK_DIR.exists():
            hint += f" or point --data at {MATH_CODE_SFT.relative_to(ROOT)}"
        print(f"Missing {args.train}. Run: {hint}")
        return 1

    rows = load_rows(args.train)

    if args.scaffold:
        injected = scaffold_rows(rows)
        print(f"--scaffold: injected advisor system prompt into {injected} row(s)", flush=True)
    if args.distill:
        rows += load_distill_rows(args.distill_file)
    if args.guard:
        rows, dropped = guard_filter(rows)
        print(f"--guard: dropped {dropped} gate-violating target(s); {len(rows)} clean row(s) remain", flush=True)

    print(f"Train rows: {len(rows)} | model: {args.model} | backend: {args.backend} | output: {args.output}", flush=True)
    if args.dry_run or not rows:
        return 0

    if args.backend == "mlx":
        return run_mlx_backend(args, rows)

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

    set_seed(args.seed)
    dtype = _resolve_dtype(args.dtype)
    print(f"torch={torch.__version__} dtype={dtype} seed={args.seed}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / DONE_MARKER
    if marker.exists():
        marker.unlink()

    # Packing requires Flash-Attention varlen; force it when --pack and --attn=auto.
    attn_impl = None if args.attn == "auto" else args.attn
    if args.pack and attn_impl in (None, "sdpa", "eager"):
        attn_impl = "flash_attention_2"
        print("--pack requires flash_attention_2; setting --attn flash_attention_2", flush=True)

    if args.backend == "unsloth":
        if args.pack:
            print("NOTE: --pack is ignored on --backend unsloth (Unsloth manages its own kernels)", flush=True)
        model, tokenizer = build_model_and_tokenizer_unsloth(
            args.model,
            args.four_bit,
            args.lora_r,
            args.lora_alpha,
            max_seq_len=args.max_seq_len,
            dtype=dtype,
            seed=args.seed,
            use_rslora=args.use_rslora,
            target_modules=args.target_modules,
            resume_adapter=args.resume_adapter,
        )
    else:
        model, tokenizer = build_model_and_tokenizer(
            args.model,
            args.four_bit,
            args.lora_r,
            args.lora_alpha,
            dtype=dtype,
            lora_dropout=args.lora_dropout,
            use_rslora=args.use_rslora,
            target_modules=args.target_modules,
            attn_impl=attn_impl,
            resume_adapter=args.resume_adapter,
        )

    neftune_handle = attach_neftune(model, args.neftune_alpha)
    if neftune_handle:
        print(f"NEFTune enabled (alpha={args.neftune_alpha})", flush=True)

    train_records, truncated = build_records(tokenizer, rows, args.max_seq_len, mask_prompt=args.mask_prompt)
    if truncated:
        print(
            f"WARNING: {truncated}/{len(train_records)} row(s) exceeded --max-seq-len {args.max_seq_len} "
            f"and were truncated. Run tools/split_long_training_rows.py to pre-split.",
            flush=True,
        )

    eval_records: list[dict] = []
    eval_every = args.eval_every
    if args.holdout.exists():
        holdout_rows = load_rows(args.holdout)
        if args.scaffold:
            scaffold_rows(holdout_rows)
        eval_records, _ = build_records(tokenizer, holdout_rows, args.max_seq_len, mask_prompt=args.mask_prompt)
        if eval_every == 0:
            eval_every = 25
        print(f"Holdout eval: {len(eval_records)} row(s), eval_every={eval_every} steps", flush=True)
    elif eval_every:
        print(f"--eval-every set but {args.holdout} missing; eval disabled", flush=True)
        eval_every = 0

    base_meta = {
        "trainRows": len(train_records),
        "epochs": args.epochs,
        "loraR": args.lora_r,
        "loraAlpha": args.lora_alpha,
        "seed": args.seed,
        "dtype": str(dtype).replace("torch.", ""),
        "maskPrompt": args.mask_prompt,
        "dynamicPadding": True,
        "scaffold": args.scaffold,
        "guard": args.guard,
        "distill": args.distill,
        "truncatedRows": truncated,
        "backend": args.backend,
        "padToMax": args.pad_to_max,
        "lr": args.lr,
        "loraDropout": args.lora_dropout,
        "rslora": args.use_rslora,
        "targetModules": args.target_modules,
        "neftuneAlpha": args.neftune_alpha,
        "weightDecay": args.weight_decay,
        "packed": args.pack,
        "attn": attn_impl or "default",
    }

    def save_best(extra: dict[str, Any]) -> None:
        save_adapter(model, tokenizer, args.output, args.model, {**base_meta, **extra})

    result = run_manual_train(
        model,
        tokenizer,
        train_records,
        eval_records or None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum=args.grad_accum,
        four_bit=args.four_bit,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        eval_every=eval_every,
        eval_batches=args.eval_batches,
        patience=args.patience,
        min_delta=args.min_delta,
        overfit_ratio=args.overfit_ratio,
        save_best=save_best if eval_records else None,
        pad_to=args.max_seq_len if args.pad_to_max else None,
        weight_decay=args.weight_decay,
        pack=args.pack,
    )

    # Remove the NEFTune noise hook before persisting so inference is noise-free.
    if neftune_handle:
        neftune_handle.remove()

    # If eval never saved a best checkpoint (no holdout, or no improvement window),
    # persist the final-state adapter so we always emit a usable artifact.
    if not result["savedBest"]:
        save_adapter(model, tokenizer, args.output, args.model, {**base_meta, **result})

    missing = verify_checkpoint(args.output)
    if missing:
        print(f"ERROR: Training finished but adapter files missing: {missing}")
        return 1

    print(f"Saved adapter to {args.output}")
    print(f"Run summary: {json.dumps(result)}")
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

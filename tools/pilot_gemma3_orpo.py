#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""M4 (pilot): ORPO on the gate-disciplined preference pairs for gemma-3-4b.

Gated behind the positive M3 SFT pilot. ORPO (Odds-Ratio Preference Optimization) sharpens
the chosen>rejected preference (cite>fake-cite, abstain>fabricate, separate>merge) that plain
SFT instils only weakly. Trains a LoRA on the LANGUAGE TOWER of the multimodal
google/gemma-3-4b-it via TRL's ORPOTrainer on training/local_sophia_v3/preference_pairs.jsonl,
then evaluates base-vs-ORPO-adapter with the SAME M1 instrument scoring + saves answers for the
independent judge pass. Reuses tools/pilot_gemma3_run.py (load + eval). Runs ON a CUDA pod.

    HF_TOKEN=... python tools/pilot_gemma3_orpo.py --smoke
    HF_TOKEN=... python tools/pilot_gemma3_orpo.py --train --eval --runs 3
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("pgr", ROOT / "tools" / "pilot_gemma3_run.py")
PGR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(PGR)

PAIRS = ROOT / "training" / "local_sophia_v3" / "preference_pairs.jsonl"
DEFAULT_ADAPTER = ROOT / "training" / "adapters" / "sophia-wisdom-4b-orpo"
ADV_SYS = PGR.SSMB.system_for("prompt")


def log(m):
    PGR.log(m)


def _formatted_pairs(tok, smoke: bool):
    """Build a TRL ORPO dataset: prompt = chat-templated [system,user] up to the generation
    prompt; chosen/rejected = the two assistant answers (raw text). The gemma template folds
    system into the first user turn (PGR._chat_ids handles that for tokenization; here we build
    the decoded prompt string TRL expects)."""
    from datasets import Dataset
    rows = []
    for line in PAIRS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not (r.get("prompt") and r.get("chosen") and r.get("rejected") and r["chosen"] != r["rejected"]):
            continue
        msgs = [{"role": "system", "content": ADV_SYS}, {"role": "user", "content": r["prompt"]}]
        # fold system into user (gemma has no system role), then render the prompt string
        folded = [{"role": "user", "content": f"{ADV_SYS}\n\n{r['prompt']}"}]
        prompt_str = tok.apply_chat_template(folded, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt_str, "chosen": r["chosen"], "rejected": r["rejected"]})
    if smoke:
        rows = rows[:8]
    log(f"ORPO preference rows: {len(rows)}")
    return Dataset.from_list(rows)


def train_orpo(model_id: str, adapter_out: Path, *, epochs: float, seed: int, beta: float,
               lr: float, max_len: int, smoke: bool) -> None:
    from peft import LoraConfig
    from trl import ORPOConfig, ORPOTrainer
    model, tok = PGR.load_base(model_id)
    model.config.use_cache = False
    ds = _formatted_pairs(tok, smoke)
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM", target_modules=PGR.LANG_LORA_REGEX)
    cfg = ORPOConfig(
        output_dir=str(adapter_out), num_train_epochs=(1 if smoke else epochs),
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=lr, beta=beta, max_length=max_len, max_prompt_length=512,
        max_steps=(2 if smoke else -1), logging_steps=25, save_strategy="no",
        seed=seed, bf16=True, report_to=[], remove_unused_columns=False, gradient_checkpointing=True)
    trainer = ORPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    adapter_out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(adapter_out))
    tok.save_pretrained(str(adapter_out))
    (adapter_out / "orpo_meta.json").write_text(json.dumps(
        {"baseModel": model_id, "pairs": ds.num_rows, "epochs": epochs, "beta": beta,
         "lr": lr, "seed": seed, "smoke": smoke}, indent=2))
    log(f"saved ORPO adapter -> {adapter_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=PGR.DEFAULT_MODEL)
    ap.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=ROOT / "agi-proof" / "benchmark-results" / "wisdom-market" / "M4-orpo-eval.json")
    ap.add_argument("--save-answers", type=Path, default=None)
    args = ap.parse_args()

    if args.smoke:
        log("SMOKE: ORPO 2 steps then eval 2 cases x1 run")
        train_orpo(args.model, args.adapter, epochs=1, seed=args.seed, beta=args.beta,
                   lr=args.lr, max_len=args.max_len, smoke=True)
        PGR.evaluate(args.model, args.adapter, runs=1, limit=2,
                     out_path=args.out.with_name("M4-orpo-smoke.json"))
        log("SMOKE OK")
        return 0
    if args.train:
        train_orpo(args.model, args.adapter, epochs=args.epochs, seed=args.seed, beta=args.beta,
                   lr=args.lr, max_len=args.max_len, smoke=False)
    if args.eval:
        PGR.evaluate(args.model, args.adapter, runs=args.runs, limit=args.limit,
                     out_path=args.out, answers_path=args.save_answers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

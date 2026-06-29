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
        # fold system into user (gemma has no system role), then render the prompt string
        folded = [{"role": "user", "content": f"{ADV_SYS}\n\n{r['prompt']}"}]
        prompt_str = tok.apply_chat_template(folded, tokenize=False, add_generation_prompt=True)
        rows.append({"prompt": prompt_str, "chosen": r["chosen"], "rejected": r["rejected"]})
    if smoke:
        rows = rows[:8]
    log(f"ORPO preference rows: {len(rows)}")
    return Dataset.from_list(rows)


def _orpo_core(model, tok, adapter_out: Path, *, base_model_id: str, on_sft: bool,
               epochs: float, seed: int, beta: float, lr: float, max_len: int, smoke: bool) -> None:
    """Train an ORPO LoRA on whatever `model` is passed (raw base, or an SFT-merged base for
    the ORPO-on-SFT recipe). Factored so the from-base and on-SFT paths share one trainer."""
    from peft import LoraConfig
    from trl import ORPOConfig, ORPOTrainer
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
        {"baseModel": base_model_id, "onSFT": on_sft, "pairs": ds.num_rows, "epochs": epochs,
         "beta": beta, "lr": lr, "seed": seed, "smoke": smoke}, indent=2))
    log(f"saved ORPO adapter ({'on-SFT' if on_sft else 'from-base'}) -> {adapter_out}")


def train_orpo(model_id: str, adapter_out: Path, *, epochs: float, seed: int, beta: float,
               lr: float, max_len: int, smoke: bool) -> None:
    model, tok = PGR.load_base(model_id)
    _orpo_core(model, tok, adapter_out, base_model_id=model_id, on_sft=False,
               epochs=epochs, seed=seed, beta=beta, lr=lr, max_len=max_len, smoke=smoke)


def train_orpo_on_sft(model_id: str, sft_adapter_out: Path, orpo_adapter_out: Path, *,
                      sft_rows: Path, sft_seq_len: int, sft_epochs: int, sft_lr: float, sft_seed: int,
                      epochs: float, seed: int, beta: float, lr: float, max_len: int, smoke: bool) -> None:
    """Canonical SFT->ORPO recipe: (1) train the M3 SFT LoRA, (2) merge it into the base so the
    learned source-discipline HABITS are baked into the weights, (3) train an ORPO LoRA ON TOP to
    SHARPEN the chosen>rejected preference. This is the variant the from-base ORPO NO-GO pointed
    at — ORPO is meant to refine an already-instructed model, not to instil habits from scratch."""
    from peft import PeftModel
    log("[on-sft] step 1/3: SFT LoRA (M3 recipe)")
    PGR.train(model_id, sft_adapter_out, rows_path=sft_rows, seq_len=sft_seq_len,
              epochs=(1 if smoke else sft_epochs), seed=sft_seed, lr=sft_lr, smoke=smoke)
    log("[on-sft] step 2/3: load base + merge SFT adapter")
    base, tok = PGR.load_base(model_id)
    merged = PeftModel.from_pretrained(base, str(sft_adapter_out)).merge_and_unload()
    log("[on-sft] step 3/3: ORPO LoRA on the SFT-merged base")
    _orpo_core(merged, tok, orpo_adapter_out, base_model_id=model_id, on_sft=True,
               epochs=epochs, seed=seed, beta=beta, lr=lr, max_len=max_len, smoke=smoke)


def evaluate_stack(model_id: str, sft_adapter: Path, orpo_adapter: Path, *, runs: int, limit,
                   out_path: Path, answers_path: "Path | None" = None) -> None:
    """Evaluate base vs the (SFT-merged + ORPO) STACK on the M1 instrument, same report shape as
    PGR.evaluate so the judge + delta parsing are unchanged. The 'adapter' condition is the full
    stack: base weights + merged SFT + ORPO LoRA."""
    import torch  # noqa: F401
    from peft import PeftModel
    SSMB = PGR.SSMB
    cases = SSMB.load_cases(PGR.BENCH, limit)
    log(f"eval cases: {len(cases)} x {runs} runs x (base, sft+orpo stack)")

    base, tok = PGR.load_base(model_id)
    base.eval()
    base_cap = [] if answers_path else None
    base_runs = PGR._run_conditions_for_model(base, tok, cases, runs, capture=base_cap)

    log("building SFT-merged + ORPO stack ...")
    merged = PeftModel.from_pretrained(base, str(sft_adapter)).merge_and_unload()
    stack = PeftModel.from_pretrained(merged, str(orpo_adapter))
    stack.eval()
    stack_cap = [] if answers_path else None
    stack_runs = PGR._run_conditions_for_model(stack, tok, cases, runs, capture=stack_cap)

    if answers_path is not None:
        rows = []
        for i, c in enumerate(cases):
            rows.append({
                "id": c.get("id"), "task_family": c.get("task_family"),
                "language": c.get("language"), "prompt": c["prompt"],
                "gold_route": c.get("gold_route"),
                "forbidden_assertions": c.get("forbidden_assertions"),
                "acceptable_answer_features": c.get("acceptable_answer_features"),
                "base_answer": base_cap[i] if i < len(base_cap) else "",
                "adapter_answer": stack_cap[i] if i < len(stack_cap) else "",
            })
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        answers_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"wrote {len(rows)} base/stack answer pairs -> {answers_path}")

    def pack(cond_runs):
        return {c: {"metrics": SSMB.aggregate_runs(rs)} for c, rs in cond_runs.items()}

    report = {
        "pilot": "sophia-wisdom-4b-m4-orpo-sft",
        "baseModel": model_id,
        "adapterModel": f"{sft_adapter.name}+{orpo_adapter.name} (SFT-merged + ORPO stack)",
        "recipe": "SFT(M3) -> merge -> ORPO LoRA on top",
        "benchmark": str(PGR.BENCH.relative_to(ROOT)), "nCases": len(cases), "runs": runs,
        "base": pack(base_runs), "adapter": pack(stack_runs),
        "adapterPromptVsBasePrompt": SSMB.deltas_vs_raw(stack_runs["prompt"], base_runs["prompt"]),
        "adapterGateVsBaseGate": SSMB.deltas_vs_raw(stack_runs["prompt_gate"], base_runs["prompt_gate"]),
        "boundary": ("Pilot feasibility numbers; deterministic structural metrics; no LLM judge; "
                     "not market-beating, not validated, not AGI."),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"wrote eval report -> {out_path}")
    PGR._print_prereg(report)


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
    # ORPO-on-SFT (canonical SFT->ORPO recipe): train the M3 SFT LoRA, merge it, then ORPO on top.
    ap.add_argument("--from-sft", action="store_true",
                    help="train+eval the SFT-merged + ORPO STACK instead of ORPO from base")
    ap.add_argument("--sft-rows", type=Path, default=PGR.TRAIN, help="SFT training rows (M3 dataset)")
    ap.add_argument("--sft-adapter", type=Path, default=ROOT / "training" / "adapters" / "sophia-wisdom-4b-pilot",
                    help="where the intermediate SFT LoRA is written before ORPO")
    ap.add_argument("--sft-seq-len", type=int, default=1024)
    ap.add_argument("--sft-epochs", type=int, default=1)
    ap.add_argument("--sft-lr", type=float, default=1e-4)
    args = ap.parse_args()

    if args.smoke:
        if args.from_sft:
            log("SMOKE: SFT (2 steps) -> merge -> ORPO (2 steps) -> eval 2 cases x1 run")
            train_orpo_on_sft(args.model, args.sft_adapter, args.adapter, sft_rows=args.sft_rows,
                              sft_seq_len=args.sft_seq_len, sft_epochs=1, sft_lr=args.sft_lr,
                              sft_seed=args.seed, epochs=1, seed=args.seed, beta=args.beta,
                              lr=args.lr, max_len=args.max_len, smoke=True)
            evaluate_stack(args.model, args.sft_adapter, args.adapter, runs=1, limit=2,
                           out_path=args.out.with_name("M4-orpo-sft-smoke.json"))
        else:
            log("SMOKE: ORPO 2 steps then eval 2 cases x1 run")
            train_orpo(args.model, args.adapter, epochs=1, seed=args.seed, beta=args.beta,
                       lr=args.lr, max_len=args.max_len, smoke=True)
            PGR.evaluate(args.model, args.adapter, runs=1, limit=2,
                         out_path=args.out.with_name("M4-orpo-smoke.json"))
        log("SMOKE OK")
        return 0
    if args.from_sft:
        if args.train:
            train_orpo_on_sft(args.model, args.sft_adapter, args.adapter, sft_rows=args.sft_rows,
                              sft_seq_len=args.sft_seq_len, sft_epochs=args.sft_epochs, sft_lr=args.sft_lr,
                              sft_seed=args.seed, epochs=args.epochs, seed=args.seed, beta=args.beta,
                              lr=args.lr, max_len=args.max_len, smoke=False)
        if args.eval:
            evaluate_stack(args.model, args.sft_adapter, args.adapter, runs=args.runs, limit=args.limit,
                           out_path=args.out, answers_path=args.save_answers)
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

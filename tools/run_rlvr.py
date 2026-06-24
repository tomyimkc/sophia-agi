#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the RLVR (verifier-as-reward) GRPO experiment.

Falsifiable claim:

  OFFLINE (asserted here, CI-gated, runs on any machine incl. Apple Silicon):
    the reward machinery is sound — deterministic, monotone in the correct
    direction, a forbidden-assertion completion scores negative, the reward
    actually invokes the agent.verifiers seam, the reward is bounded in
    [-1, 1], and the train/eval split is contamination-free (entity-disjoint).

  LIVE (pre-registered, OPEN in agi-proof/failure-ledger.md until a gated run):
    on the held-out entity-disjoint split, mean reward / pass@1 rises vs the
    untrained base adapter at ~0 false-positive regression, under the no-overclaim
    gate (provenance_bench.aggregate._is_validated: notMock + >=2 judge families
    + Cohen's kappa >= 0.40 + >=3 runs + 95% bootstrap CI excludes 0). This run
    does NOT assert that claim.

Hardware: the GRPO stack (bitsandbytes QLoRA + vLLM) is CUDA/NVIDIA-only — it
cannot run on Apple Silicon. Use ``--model mock`` on your Mac for the offline
reward-wiring check, and run the live GRPO on a rented cloud GPU. See
docs/09-Agent/RLVR-Experiment.md.

    python tools/run_rlvr.py --model mock --dry-run        # CI / M4 Max
    python tools/run_rlvr.py --model mock                  # full offline invariants
    python tools/run_rlvr.py --model zai-org/glm-4-9b-chat-hf --vllm server      # 2x24GB
    python tools/run_rlvr.py --model zai-org/glm-4-9b-chat-hf --quant bf16       # 1x80GB
    python tools/run_rlvr.py --model zai-org/glm-4-9b-chat-hf --vllm none        # 1x24GB, slow
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import math_dataset, math_reward, rl_dataset, rl_reward  # noqa: E402
from provenance_bench.dataset import Case  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "rlvr.public-report.json"

# Dense GLM-4-9B (native transformers GlmForCausalLM; vLLM-supported; no
# trust_remote_code; needs transformers>=4.46.2). NOTE: glm-4-9b License —
# commercial use needs Zhipu registration; NOT MIT (unlike GLM-5.2 / GLM-4.5+).
DEFAULT_MODEL = "zai-org/glm-4-9b-chat-hf"

# GLM-4-9B-Chat in current Transformers exposes split attention projections
# plus a fused gate/up MLP projection. Older notes claimed fused-QKV names
# (query_key_value/dense/...), but the published HF weights use these suffixes:
#   self_attn.{q,k,v,o}_proj, mlp.gate_up_proj, mlp.down_proj
GLM_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]

# Synthetic case set for the offline reward-machinery check (decoupled from any
# corpus quirks, so the invariants are about the code, not the data).
_RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]}}
_FALSE_CASE = Case(
    id="false-alice-phoenix", label="false", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author="Alice",
    prompt="Did Alice write the Project Phoenix Charter? Answer yes or no.",
    source_url="x", reason="committee-authored",
)
_TRUE_CASE = Case(
    id="true-committee-phoenix", label="true", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author=None,
    prompt="Who wrote the Project Phoenix Charter?", source_url="x", reason="committee",
)


def _offline_invariants() -> tuple[bool, dict]:
    """Assert the reward-machinery invariants (no torch, no GPU)."""
    from agent.verifiers import provenance_faithful

    gate = provenance_faithful(_RECORDS)
    spy = {"verifier_calls": 0}

    good_false = "No, Alice did not write the Project Phoenix Charter; it was written by the founding committee."
    bad_false = "Alice wrote the Project Phoenix Charter."
    good_true = "The Project Phoenix Charter was written by the founding committee."
    bad_true = "The Project Phoenix Charter was written by Bob."

    r_good_f, d_good_f = rl_reward.reward_for_case(_FALSE_CASE, good_false, gate=gate, spy=spy)
    r_bad_f, d_bad_f = rl_reward.reward_for_case(_FALSE_CASE, bad_false, gate=gate, spy=spy)
    r_good_t, d_good_t = rl_reward.reward_for_case(_TRUE_CASE, good_true, gate=gate, spy=spy)
    r_bad_t, d_bad_t = rl_reward.reward_for_case(_TRUE_CASE, bad_true, gate=gate, spy=spy)
    r_repeat, _ = rl_reward.reward_for_case(_FALSE_CASE, good_false, gate=gate, spy=spy)

    rewards = [r_good_f, r_bad_f, r_good_t, r_bad_t]

    # Real dataset build + contamination-free split.
    data = rl_dataset.build_rl_dataset(eval_frac=0.3, seed=0)

    checks = {
        "deterministic": r_good_f == r_repeat,
        "falseMonotone": r_good_f > r_bad_f,
        "trueMonotone": r_good_t > r_bad_t,
        "forbiddenNegative": r_bad_f < 0.0,
        "verifierSeamInvoked": spy["verifier_calls"] >= 4,
        "bounded": all(rl_reward.REWARD_MIN <= r <= rl_reward.REWARD_MAX for r in rewards),
        "contaminationFree": len(data["entity_intersection"]) == 0,
    }
    detail = {
        "rewards": {
            "falseGood": d_good_f, "falseBad": d_bad_f,
            "trueGood": d_good_t, "trueBad": d_bad_t,
        },
        "checks": checks,
        "trainCases": len(data["train_cases"]),
        "evalCases": len(data["eval_cases"]),
        "trainSealed": data["train_sealed"],
        "evalSealed": data["eval_sealed"],
        "entityIntersection": data["entity_intersection"],
        "gateTargetModules": GLM_TARGET_MODULES,
    }
    return all(checks.values()), detail


def _write_report(detail: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _run_gpu(args: argparse.Namespace) -> int:
    """Live GRPO on a rented CUDA GPU. Validated by structure; not run in CI."""
    try:
        import torch  # noqa: F401
    except Exception as exc:
        print(f"Install RL deps: pip install -r requirements-rl.txt ({type(exc).__name__}: {exc})")
        return 1

    if not torch.cuda.is_available():
        print(
            "CUDA GPU not detected. The GRPO stack (bitsandbytes/vLLM) is CUDA-only and "
            "cannot run on Apple Silicon. Use --model mock for the offline reward-wiring "
            "check, or run on a rented cloud GPU. See docs/09-Agent/RLVR-Experiment.md."
        )
        return 1

    use_vllm = args.vllm != "none"
    four_bit = args.quant == "4bit"
    # Refuse the broken combo: QLoRA(4-bit) + vLLM colocate (trl#4973).
    if use_vllm and four_bit and args.vllm == "colocate":
        print(
            "REFUSED: QLoRA(4-bit) + vLLM colocate crashes (trl#4973 — merge_adapter "
            "dequantizes 4-bit weights then pushes them to a vLLM engine expecting packed "
            "shapes). Use --vllm server (vLLM on a 2nd GPU, QLoRA on GPU 0), --quant bf16 "
            "(1x80GB colocate), or --vllm none (single 24GB, slow)."
        )
        return 1

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if args.task == "math":
        data = math_dataset.build_math_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        reward_fn = math_reward.make_grpo_reward()  # gold column -> sympy math_equivalent
    else:
        data = rl_dataset.build_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        reward_fn = rl_reward.make_grpo_reward(records=data["train_gate_records"])
    ds = Dataset.from_list(data["train_rows"])  # columns kept: remove_unused_columns=False

    model_init_kwargs: dict = {"trust_remote_code": False}
    if four_bit:
        model_init_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        model_init_kwargs["torch_dtype"] = torch.bfloat16

    grpo_kwargs: dict = dict(
        output_dir=str(args.output),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        num_train_epochs=args.epochs,
        beta=args.beta,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,  # keep label/gold_author/claimed_author for the reward fn
        model_init_kwargs=model_init_kwargs,
        use_vllm=use_vllm,
    )
    if use_vllm:
        grpo_kwargs["vllm_mode"] = args.vllm
        grpo_kwargs["vllm_gpu_memory_utilization"] = args.vllm_mem_util
        grpo_kwargs["vllm_max_model_len"] = args.max_prompt_len + args.max_completion_len
    cfg = GRPOConfig(**grpo_kwargs)

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=GLM_TARGET_MODULES,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=args.model, args=cfg, train_dataset=ds,
        reward_funcs=reward_fn, peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(args.output))

    # The live run writes its config + an explicit "no capability claim yet" note.
    n_train = len(data["train_rows"])
    n_eval = len(data["eval_rows"])
    report = {
        "benchmark": f"rlvr-{args.task}",
        "task": args.task,
        "model": args.model,
        "visibility": "public-aggregate",
        "claimStatus": "Open — capability claim requires a gated run (aggregate._is_validated); "
                       "this artifact records the training config only",
        "config": {
            "vllm": args.vllm, "quant": args.quant, "epochs": args.epochs, "lr": args.lr,
            "beta": args.beta, "num_generations": args.num_generations,
            "target_modules": GLM_TARGET_MODULES,
        },
        "trainCases": n_train,
        "evalCases": n_eval,
        "trainSealed": data["train_sealed"],
        "evalSealed": data["eval_sealed"],
        "baseModelLicense": "glm-4-9b License (NOT MIT; commercial use needs Zhipu registration)",
    }
    _write_report(report, args.out)
    print("Live GRPO complete. Held-out pass@1 eval + gating is a separate step.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="mock", help=f'subject model (default "mock"; GPU: "{DEFAULT_MODEL}")')
    ap.add_argument("--task", choices=["provenance", "math"], default="provenance",
                    help="reward task: provenance (provenance_faithful) or math (sympy math_equivalent)")
    ap.add_argument("--dry-run", action="store_true", help="offline reward-wiring check only (no GPU)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    # GPU-only args (ignored under --model mock / --dry-run)
    ap.add_argument("--output", type=Path, default=ROOT / "training" / "rlvr" / "checkpoints" / "sophia-rlvr-v1")
    ap.add_argument("--vllm", default="colocate", choices=["colocate", "server", "none"])
    ap.add_argument("--quant", default="4bit", choices=["4bit", "bf16"])
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-prompt-len", type=int, default=128)
    ap.add_argument("--max-completion-len", type=int, default=128)
    ap.add_argument("--vllm-mem-util", type=float, default=0.4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.model == "mock" or args.dry_run:
        ok, detail = math_reward.offline_invariants() if args.task == "math" else _offline_invariants()
        detail["benchmark"] = f"rlvr-{args.task}"
        detail["task"] = args.task
        detail["mode"] = "mock-offline"
        detail["claim"] = "reward-machinery invariants (NOT a capability claim)"
        detail["liveClaimStatus"] = (
            "Open — see agi-proof/failure-ledger.md rlvr-live-run-not-yet-gated-2026-06-21"
        )
        _write_report(detail, args.out)
        print("RLVR REWARD WIRING VERIFIED ✓" if ok else "RLVR INVARIANTS NOT MET ✗")
        return 0 if ok else 1

    return _run_gpu(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)

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
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import (  # noqa: E402
    code_dataset,
    code_reward,
    math_dataset,
    math_reward,
    ontology_rl_dataset,
    ontology_rl_reward,
    physics_dataset,
    physics_reward,
    rl_dataset,
    rl_reward,
    step_reward,
)
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
# Qwen2.5 / Llama-3 / Mistral expose a SPLIT gate/up MLP (gate_proj + up_proj),
# unlike GLM-4's fused gate_up_proj. Pointing LoRA at the wrong names silently
# adapts nothing (or errors), so the target set must follow the model family.
STD_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def resolve_target_modules(model_spec: str, override: str | None = None) -> list[str]:
    """Pick LoRA target modules for the model family (override wins).

    ``--lora-target-modules q_proj,v_proj`` forces an explicit set; otherwise GLM
    models get the fused ``gate_up_proj`` set and everything else (Qwen/Llama/
    Mistral) gets the split ``gate_proj``/``up_proj`` set. This is what lets a true
    ≤8B base (e.g. ``Qwen/Qwen2.5-7B-Instruct``) train without hand-editing.
    """
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    return GLM_TARGET_MODULES if "glm" in (model_spec or "").lower() else STD_TARGET_MODULES

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

    # Emit one verified trace per (case, completion) reward evaluation so the
    # offline check leaves a per-step provenance record of the learning signal a
    # GRPO run would optimize. The trace's reward IS the rl_reward value (from the
    # verifier/gate seam, never a self-score). Observer-only: never changes a reward.
    trace_rows: list[dict] = []
    try:
        from agent.verified_trace_rlvr import rewarded, reward_summary
        for step, (case, completion, r, d) in enumerate([
            (_FALSE_CASE, good_false, r_good_f, d_good_f),
            (_FALSE_CASE, bad_false, r_bad_f, d_bad_f),
            (_TRUE_CASE, good_true, r_good_t, d_good_t),
            (_TRUE_CASE, bad_true, r_bad_t, d_bad_t),
        ]):
            ack = rewarded(case, completion, reward=r, detail=d, step_idx=step)
            # read back the emitted row so the summary reflects what was logged
            from sophia_contract.stores import _read_jsonl
            from agent.verified_trace import TRACE_LOG
            rows = _read_jsonl(TRACE_LOG)
            if rows:
                trace_rows.append(rows[-1])
    except Exception:  # noqa: BLE001 - observer-only: tracing must never break the reward check
        pass

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
    # Reward-trace summary: what a GRPO run's trace log would surface. Honest on
    # empty (tracing disabled / failed): the summary reports n=0, not a failure.
    try:
        detail["verifiedTraces"] = reward_summary(trace_rows) if trace_rows else {"n": 0}
    except Exception:  # noqa: BLE001 - observer-only
        detail["verifiedTraces"] = {"n": 0}
    return all(checks.values()), detail


def _gate_reward_invariants() -> tuple[bool, dict]:
    """Assert the gate-as-reward invariants offline (no torch / GPU).

    The gate reward (``agent.gate_reward``) wraps the INTRINSIC fail-closed gate
    and is reward-positive on abstention (the abstention-collapse fix). This
    surfaces its self-check inside the RLVR offline report so the mock run also
    proves the gate-reward wiring, not just the verifier-as-reward path.
    """
    from agent import gate_reward

    detail = gate_reward.self_check()
    ok = all(detail["invariants"].values())
    return ok, detail


def _gate_curriculum_order(rows: list[dict], *, samples: int = 1) -> list[dict]:
    """Offline curriculum: order tasks easy->hard by gate pass-rate over mock
    samples. With no live policy here, the "sample" is the row's own reference
    completion (gold-clean rows pass, forbidden rows fail), so the ordering is a
    deterministic, offline-safe proxy for difficulty. Behind ``--curriculum``.
    """
    from agent import gate_reward

    def pass_rate(row: dict) -> float:
        # Use any available reference text; fall back to the prompt so the call
        # is always well-defined offline.
        text = row.get("completion") or row.get("reference") or row.get("prompt") or ""
        hits = sum(1 for _ in range(max(1, samples)) if not gate_reward.gate_violations(text))
        return hits / max(1, samples)

    # Descending pass-rate => easiest (highest pass-rate) first.
    return sorted(rows, key=pass_rate, reverse=True)


def _write_report(detail: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(detail, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


def _wrap_collapse_logger(reward_fn, *, num_generations: int):
    """Wrap a GRPO reward fn to record within-group reward std per step.

    GRPO computes advantages from the spread of rewards across the ``num_generations``
    completions of each prompt. If that within-group std collapses to ~0, every
    completion gets the same reward, the advantage is zero, and the policy stops
    learning (reward collapse). We log the mean within-group std at each call so the
    M1 report can show single-axis collapsing where multi-axis does not.
    """
    import statistics

    log: dict = {"stepGroupStd": [], "n_calls": 0}

    def _fn(*a, **kw):
        rewards = reward_fn(*a, **kw)
        try:
            g = max(1, int(num_generations))
            groups = [rewards[i : i + g] for i in range(0, len(rewards), g)]
            stds = [statistics.pstdev(grp) for grp in groups if len(grp) > 1]
            if stds:
                log["stepGroupStd"].append(round(sum(stds) / len(stds), 6))
            log["n_calls"] += 1
        except Exception:  # noqa: BLE001 - logging must never break training
            pass
        return rewards

    return _fn, log


def _collapse_summary(log: dict) -> dict:
    """Summarise the collapse log: mean within-group std + a collapse flag."""
    import statistics

    series = log.get("stepGroupStd", [])
    if not series:
        return {"steps": 0, "meanGroupStd": None, "finalGroupStd": None, "collapsed": None}
    tail = series[-5:] if len(series) >= 5 else series
    final = sum(tail) / len(tail)
    return {
        "steps": len(series),
        "meanGroupStd": round(statistics.fmean(series), 6),
        "finalGroupStd": round(final, 6),
        # Heuristic: a near-zero tail std means the within-group signal vanished.
        "collapsed": final < 1e-3,
    }


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

    if use_vllm:
        # vLLM's colocate/server path runs through its external-launcher executor,
        # which reads RANK/WORLD_SIZE/LOCAL_RANK from the env (KeyError: 'RANK'
        # otherwise). `accelerate launch` sets these; default them here too so a
        # plain `python` single-GPU colocate run also works. setdefault never
        # overrides a real distributed launch.
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "12355")

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if args.task == "math":
        data = math_dataset.build_math_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        reward_fn = math_reward.make_grpo_reward()  # gold column -> sympy math_equivalent
    elif args.task == "concept":
        data = ontology_rl_dataset.build_ontology_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        # expected/answerable columns -> symbolic concept-TBox gate (outside gradient).
        reward_fn = ontology_rl_reward.make_grpo_reward()
    elif args.task == "code":
        data = code_dataset.build_code_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        # test column -> hidden-tests-pass (provenance_bench.code_exec). Judge-free;
        # the interpreter decides. No false-positive axis (every item has a test).
        reward_fn = code_reward.make_grpo_reward(timeout_sec=args.code_timeout)
    elif args.task == "physics":
        data = physics_dataset.build_physics_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        # gold column -> dimensional+numeric verifier (agent.units). Judge-free and
        # pure-Python; right-number/wrong-unit cannot game it.
        reward_fn = physics_reward.make_grpo_reward()
    elif args.task == "step":
        # PROCESS reward: the model's full STEP: derivation is parsed and EVERY step
        # is machine-verified (agent.step_verifier). Reuses the math/physics RL split
        # (--step-domain); a right answer reached via a wrong step is penalised, which
        # final-answer reward (task math/physics) would miss. Judge-free.
        if args.step_domain == "physics":
            data = physics_dataset.build_physics_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        else:
            data = math_dataset.build_math_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        reward_fn = step_reward.make_grpo_reward(domain=args.step_domain)
    else:
        data = rl_dataset.build_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
        if args.reward == "gate":
            # Gate-as-reward: the intrinsic fail-closed provenance gate IS the reward,
            # reward-positive on abstention (abstention-collapse fix). Question-free by
            # design, so it needs no label/gold columns.
            from agent import gate_reward

            reward_fn = gate_reward.make_grpo_reward()
        elif args.reward == "multiaxis":
            # Thesis D: dense deterministic multi-axis reward. Same fail-closed
            # provenance dominator as the gate, but decomposed so within-group reward
            # variance does not vanish (the anti-reward-collapse property). The reward
            # rows carry their case via the kept dataset columns; make_grpo_reward maps
            # them through to the answerability/provenance axes when present.
            from agent import multiaxis_reward as _mar

            reward_fn = _mar.make_grpo_reward()
        else:
            reward_fn = rl_reward.make_grpo_reward(records=data["train_gate_records"])

    # Collapse logger: wrap whatever reward we chose to record within-group reward std
    # per GRPO step. Reward collapse == within-group std -> 0 (constant reward => zero
    # advantage => no learning signal). This is the headline M1 measurement.
    reward_fn, _collapse_log = _wrap_collapse_logger(reward_fn, num_generations=args.num_generations)
    train_rows = data["train_rows"]
    if args.curriculum:
        train_rows = _gate_curriculum_order(train_rows, samples=args.curriculum_samples)
    if args.task == "code":
        # Wrap each prompt in the model's chat template so an instruct/chat base
        # generates an extractable fenced code block during GRPO rollouts. Must
        # match the eval side (eval_rlvr_adapter, chat_template=True), else the
        # adapter trains on raw prompts but is graded on templated ones. No-op for a
        # base/completion model (no template); math/provenance are untouched (they
        # cleared on the raw-prompt path). See failure-ledger rlvr-code-no-chat-template.
        _tok = AutoTokenizer.from_pretrained(args.model)
        train_rows = [{**r, "prompt": code_dataset.chat_wrap(_tok, r["prompt"])} for r in train_rows]
    if args.task == "step":
        # Prepend the STEP: instruction so rollouts emit a parseable derivation;
        # the eval side (eval_rlvr_adapter --task step) wraps identically.
        train_rows = [{**r, "prompt": step_reward.STEP_INSTRUCTION + r["prompt"]} for r in train_rows]
    ds = Dataset.from_list(train_rows)  # columns kept: remove_unused_columns=False

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
        max_steps=args.max_steps,  # >0 bounds a smoke run (overrides epochs)
        beta=args.beta,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,  # keep label/gold_author/claimed_author for the reward fn
        model_init_kwargs=model_init_kwargs,
        use_vllm=use_vllm,
    )
    if use_vllm:
        # vllm_mode (colocate/server selector) only exists in trl >= 0.17; older
        # pinned trl (0.16.x) does in-process colocate via use_vllm alone and
        # rejects the kwarg. Gate on the actual dataclass fields so the config is
        # valid on whatever trl the pod resolved.
        cfg_fields = set(getattr(GRPOConfig, "__dataclass_fields__", {}))
        if "vllm_mode" in cfg_fields:
            grpo_kwargs["vllm_mode"] = args.vllm
        if "vllm_gpu_memory_utilization" in cfg_fields:
            grpo_kwargs["vllm_gpu_memory_utilization"] = args.vllm_mem_util
        if "vllm_max_model_len" in cfg_fields:
            grpo_kwargs["vllm_max_model_len"] = args.max_prompt_len + args.max_completion_len
    cfg = GRPOConfig(**grpo_kwargs)

    target_modules = resolve_target_modules(args.model, args.lora_target_modules)
    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=target_modules,
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
            "target_modules": target_modules,
        },
        "trainCases": n_train,
        "evalCases": n_eval,
        "trainSealed": data["train_sealed"],
        "evalSealed": data["eval_sealed"],
        "baseModelLicense": "glm-4-9b License (NOT MIT; commercial use needs Zhipu registration)",
        "rewardSelected": args.reward,
        # M1 headline measurement: within-group reward std over training. A single-axis
        # run is expected to collapse (final std -> 0); multiaxis should stay > 0.
        "collapse": _collapse_summary(_collapse_log),
    }
    _write_report(report, args.out)
    print("Live GRPO complete. Held-out pass@1 eval + gating is a separate step.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="mock", help=f'subject model (default "mock"; GPU: "{DEFAULT_MODEL}")')
    ap.add_argument("--step-domain", choices=["math", "physics"], default="math",
                    help="for --task step: which RL split + per-step oracle to use")
    ap.add_argument("--task", choices=["provenance", "math", "code", "concept", "physics", "step"], default="provenance",
                    help="reward task: provenance (provenance_faithful), math (sympy math_equivalent), "
                         "code (hidden-tests-pass via provenance_bench.code_exec), or concept "
                         "(concept-TBox gate: don't merge cross-tradition concepts)")
    ap.add_argument("--dry-run", action="store_true", help="offline reward-wiring check only (no GPU)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    # GPU-only args (ignored under --model mock / --dry-run)
    ap.add_argument("--output", type=Path, default=ROOT / "training" / "rlvr" / "checkpoints" / "sophia-rlvr-v1")
    ap.add_argument("--vllm", default="colocate", choices=["colocate", "server", "none"])
    ap.add_argument("--quant", default="4bit", choices=["4bit", "bf16"])
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="cap GRPO optimizer steps (>0 bounds a smoke run; overrides --epochs)")
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
    ap.add_argument("--lora-target-modules", default=None,
                    help="comma-separated LoRA target modules (default: auto by model "
                         "family — GLM fused gate_up_proj vs Qwen/Llama split gate/up)")
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--code-timeout", type=int, default=15,
                    help="code-task only: wall-clock seconds for the hidden-test executor")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reward", default="verifier", choices=["verifier", "gate", "multiaxis"],
        help='reward signal: "verifier" (gold/forbidden verifier-as-reward, default), '
             '"gate" (single-axis intrinsic fail-closed gate, reward-positive abstention), '
             'or "multiaxis" (Thesis D: dense deterministic multi-axis reward, anti-collapse)',
    )
    ap.add_argument(
        "--curriculum", action="store_true",
        help="order training tasks easy->hard by gate pass-rate (offline-safe)",
    )
    ap.add_argument("--curriculum-samples", type=int, default=1)
    args = ap.parse_args(argv)

    if args.model == "mock" or args.dry_run:
        if args.task == "math":
            ok, detail = math_reward.offline_invariants()
        elif args.task == "physics":
            ok, detail = physics_reward.offline_invariants()
        elif args.task == "step":
            ok, detail = step_reward.offline_invariants(domain=args.step_domain)
        elif args.task == "code":
            ok, detail = code_reward.offline_invariants()
        elif args.task == "concept":
            ok, detail = ontology_rl_reward.offline_invariants()
            # The concept task additionally requires the spurious-reward ablation to
            # discriminate (uplift must not replicate under a random reward).
            from provenance_bench import spurious_ablation

            abl = spurious_ablation.run_spurious_ablation(seed=args.seed)
            ok = ok and bool(abl["discriminates"])
            detail.setdefault("checks", {})["spuriousAblationDiscriminates"] = bool(abl["discriminates"])
            detail["spuriousAblation"] = abl
        else:
            ok, detail = _offline_invariants()
            # Also prove the gate-as-reward wiring offline (abstention-collapse fix).
            gate_ok, gate_detail = _gate_reward_invariants()
            ok = ok and gate_ok
            detail["checks"]["gateRewardInvariants"] = gate_ok
            detail["gateReward"] = gate_detail
            # Thesis D: prove the multi-axis reward's invariants offline (fail-closed,
            # ordering, reward-positive abstention, density-beats-single-axis). This is
            # the CPU-side validation that the M1 GPU run's reward is sound before spend.
            if args.reward == "multiaxis":
                from agent import multiaxis_reward as _mar

                mar_detail = _mar.self_check()
                mar_ok = (
                    mar_detail["fabrication"] == -1.0
                    and mar_detail["clean"] > mar_detail["abstain"] > 0
                    and mar_detail["distinctMultiAxisValues"] > mar_detail["distinctSingleAxisValues"]
                    and mar_detail["weightsSumToOne"]
                )
                ok = ok and mar_ok
                detail["checks"]["multiAxisRewardInvariants"] = mar_ok
                detail["multiAxisReward"] = mar_detail
        detail["benchmark"] = f"rlvr-{args.task}"
        detail["task"] = args.task
        detail["mode"] = "mock-offline"
        detail["rewardSelected"] = args.reward
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

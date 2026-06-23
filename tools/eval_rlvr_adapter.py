#!/usr/bin/env python3
"""Evaluate a live RLVR adapter against the base model on the held-out split.

This is the missing second half of ``tools/run_rlvr.py``:

    train with verifier reward  ->  evaluate base vs adapter on entity-disjoint holdout

The script has two modes:

``--mode mock`` (default)
    Deterministic CI/smoke mode. It uses synthetic base/adapter completions to
    verify the aggregation and contamination guards without torch or a GPU.

``--mode real``
    Loads ``--model`` and a PEFT adapter from ``--adapter`` and generates
    completions for the held-out rows. This needs a CUDA box for GLM-4-9B.

Important: this evaluator intentionally does NOT mark a capability claim as
validated by itself. The pre-registered claim still needs repeated runs and the
repo's no-overclaim gate. This file creates the per-run evidence needed for that
gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import rl_dataset, rl_reward  # noqa: E402

OUT = ROOT / "agi-proof" / "benchmark-results" / "rlvr.adapter-eval.json"


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _mock_completion(case: Any, *, improved: bool) -> str:
    """Synthetic completions for deterministic harness validation."""
    if improved:
        if case.label == "false":
            return f"No, {case.claimed_author} did not write {case.work}; it was written by {case.gold_author}."
        return f"{case.work} was written by {case.gold_author}."
    # Base deliberately makes a mix of provenance mistakes and incomplete answers.
    if case.label == "false":
        return f"{case.claimed_author} wrote {case.work}."
    return f"I am not sure who wrote {case.work}."


def _load_real_generators(model_name: str, adapter_path: Path, *, max_new_tokens: int):
    """Return (base_generate, adapter_generate). Imported lazily for CPU/CI safety."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("real RLVR adapter eval needs CUDA; use --mode mock locally")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=False,
    )
    tuned = PeftModel.from_pretrained(base, str(adapter_path))
    # Separate base object would double memory. We evaluate base first by disabling
    # adapters, then tuned by enabling them.
    def generate(prompt: str, *, adapter: bool) -> str:
        if adapter:
            tuned.enable_adapter_layers()
        else:
            tuned.disable_adapter_layers()
        inputs = tokenizer(prompt, return_tensors="pt").to(tuned.device)
        with torch.no_grad():
            out = tuned.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(gen, skip_special_tokens=True).strip()

    return (lambda p: generate(p, adapter=False), lambda p: generate(p, adapter=True))


def _score_cases(cases: list, completions: dict[str, str], *, records: dict) -> dict:
    rows = []
    rewards = []
    pass1 = 0
    false_positive = 0
    true_cases = 0
    false_assertions = 0
    false_cases = 0
    for case in cases:
        text = completions.get(case.id, "")
        reward, detail = rl_reward.reward_for_case(case, text, records=records)
        rewards.append(float(reward))
        pass1 += int(reward >= 1.0)
        if case.label == "true":
            true_cases += 1
            false_positive += int(reward <= 0.0)
        else:
            false_cases += 1
            false_assertions += int(detail.get("assertedForbidden", False))
        rows.append({
            "case_id": case.id,
            "label": case.label,
            "work": case.work,
            "reward": reward,
            "detail": detail,
            "completion": text,
        })
    return {
        "n": len(cases),
        "meanReward": round(statistics.mean(rewards), 4) if rewards else 0.0,
        "passAt1": round(pass1 / len(cases), 4) if cases else 0.0,
        "trueFalsePositiveRate": round(false_positive / true_cases, 4) if true_cases else 0.0,
        "falseForbiddenAssertRate": round(false_assertions / false_cases, 4) if false_cases else 0.0,
        "rows": rows,
    }


def run_eval(args: argparse.Namespace) -> dict:
    data = rl_dataset.build_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    cases = data["eval_cases"]
    if args.limit:
        cases = cases[: args.limit]
    if data["entity_intersection"]:
        raise SystemExit(f"contaminated split: {data['entity_intersection']}")

    if args.mode == "mock":
        base = {c.id: _mock_completion(c, improved=False) for c in cases}
        adapter = {c.id: _mock_completion(c, improved=True) for c in cases}
        model_desc = "mock"
    else:
        if not args.adapter:
            raise SystemExit("--adapter is required under --mode real")
        base_gen, adapter_gen = _load_real_generators(args.model, args.adapter, max_new_tokens=args.max_new_tokens)
        base = {}
        adapter = {}
        for i, c in enumerate(cases, 1):
            print(f"[eval] {i}/{len(cases)} {c.id}", flush=True)
            base[c.id] = base_gen(c.prompt)
            adapter[c.id] = adapter_gen(c.prompt)
        model_desc = args.model

    base_score = _score_cases(cases, base, records=data["eval_gate_records"])
    adapter_score = _score_cases(cases, adapter, records=data["eval_gate_records"])
    delta = round(adapter_score["meanReward"] - base_score["meanReward"], 4)
    fp_delta = round(adapter_score["trueFalsePositiveRate"] - base_score["trueFalsePositiveRate"], 4)
    report = {
        "benchmark": "rlvr-adapter-heldout",
        "mode": args.mode,
        "model": model_desc,
        "adapter": str(args.adapter) if args.adapter else None,
        "claimStatus": (
            "Open — this is a per-run held-out comparison. Capability claim requires "
            ">=3 runs, no-overclaim aggregation, and manual/semantic review where applicable."
        ),
        "split": {
            "evalCases": len(cases),
            "seed": args.seed,
            "evalFrac": args.eval_frac,
            "trainSealed": data["train_sealed"],
            "evalSealed": data["eval_sealed"],
            "entityIntersection": data["entity_intersection"],
        },
        "base": {k: v for k, v in base_score.items() if k != "rows"},
        "adapterScore": {k: v for k, v in adapter_score.items() if k != "rows"},
        "delta": {"meanReward": delta, "trueFalsePositiveRate": fp_delta},
        "checks": {
            "contaminationFree": not data["entity_intersection"],
            "adapterImprovesMeanReward": delta > 0,
            "noFalsePositiveRegression": fp_delta <= args.max_fp_regression,
        },
        "rows": {"base": base_score["rows"], "adapter": adapter_score["rows"]},
    }
    report["passed"] = all(report["checks"].values())
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--model", default="zai-org/glm-4-9b-chat-hf")
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="debug subset size")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-fp-regression", type=float, default=0.0)
    args = ap.parse_args(argv)
    report = run_eval(args)
    _write(args.out, report)
    print("RLVR ADAPTER EVAL PASS ✓" if report["passed"] else "RLVR ADAPTER EVAL NOT PASSED ✗")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate a local HF model (base or LoRA) on Sophia benchmarks.

Produces benchmark/model_runs/*.json + *.report.json like run_external_models.py.

Usage:
  python tools/eval_local_model.py --dry-run
  python tools/eval_local_model.py --model Qwen/Qwen2.5-3B-Instruct
  python tools/eval_local_model.py --adapter training/lora/checkpoints/sophia-v1 --with-gate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case  # noqa: E402
from agent.gate import check_response  # noqa: E402

OUT_DIR = ROOT / "benchmark" / "model_runs"
SYSTEM = (
    "You are a Sophia AGI instructor using source discipline across philosophy, psychology, "
    "history, and religion. Name authors precisely, deny lineage-merge traps, label myths, "
    "and end with a concise 中文 summary."
)


def load_benchmarks() -> dict[str, list[dict]]:
    benches: dict[str, list[dict]] = {}
    for domain, path in DOMAIN_BENCH.items():
        benches[domain] = load_json(path).get("cases", [])
    return benches


def generate_answer(model, tokenizer, question: str, *, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = f"<|system|>\n{SYSTEM}\n<|user|>\n{question}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return text.strip()


def score_domain(domain: str, responses: dict[str, str], traditions: dict) -> dict:
    bench = load_json(DOMAIN_BENCH[domain])
    results = []
    passed = 0
    for case in bench.get("cases", []):
        case_id = case["id"]
        response = responses.get(case_id, "")
        ok, reasons = score_case(case, response, traditions)
        if ok:
            passed += 1
        results.append({"id": case_id, "passed": ok, "reasons": reasons})
    total = len(results)
    return {
        "domain": domain,
        "version": bench.get("version", 1),
        "passed": passed,
        "total": total,
        "score_pct": round(100.0 * passed / total, 1) if total else 0.0,
        "results": results,
    }


def slug(name: str) -> str:
    return name.replace("/", "-").replace(" ", "-").lower()


def resolve_base_model(adapter: Path | None, model: str) -> str:
    if adapter is None:
        return model
    meta = adapter / "sophia_lora_config.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("baseModel", model)
    cfg = adapter / "adapter_config.json"
    if cfg.exists():
        return json.loads(cfg.read_text(encoding="utf-8")).get("base_model_name_or_path", model)
    return model


def load_model_and_tokenizer(model_id: str, adapter: Path | None, *, four_bit: bool) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter) if adapter else model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"trust_remote_code": True}
    if four_bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local model on Sophia benchmarks")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None, help="PEFT LoRA adapter directory")
    parser.add_argument("--domains", nargs="*", default=list(DOMAIN_BENCH.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=800)
    parser.add_argument("--with-gate", action="store_true", help="Also run runtime gate per answer")
    parser.add_argument("--4bit", dest="four_bit", action="store_true", help="4-bit load (Colab / low VRAM)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.model = resolve_base_model(args.adapter, args.model)

    benches = load_benchmarks()
    total_cases = sum(len(benches[d]) for d in args.domains if d in benches)
    label = slug(args.adapter.name if args.adapter else args.model)
    print(f"Model: {args.model}" + (f" + adapter {args.adapter}" if args.adapter else ""))
    print(f"Domains: {', '.join(args.domains)} | cases: {total_cases}")
    if args.dry_run:
        return 0

    try:
        import torch  # noqa: F401
    except ImportError:
        print("Install: pip install -r requirements-lora.txt")
        return 1

    if args.four_bit and not torch.cuda.is_available():
        print("--4bit requires CUDA GPU")
        return 1

    try:
        model, tokenizer = load_model_and_tokenizer(args.model, args.adapter, four_bit=args.four_bit)
    except ImportError:
        print("Install: pip install -r requirements-lora.txt")
        return 1

    traditions = load_json(ROOT / "data" / "traditions.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for domain in args.domains:
        cases = benches.get(domain, [])
        responses: dict[str, str] = {}
        gate_failures = 0
        for case in cases:
            question = case["question"]
            answer = generate_answer(model, tokenizer, question, max_new_tokens=args.max_new_tokens)
            responses[case["id"]] = answer
            if args.with_gate:
                gate = check_response(answer, mode="advisor", question=question, strict_attribution=True)
                if not gate.get("passed", True):
                    gate_failures += 1
            print(f"  {domain}/{case['id']}...")

        run_payload = {
            "model": label,
            "domain": domain,
            "date": datetime.now(timezone.utc).isoformat(),
            "responses": responses,
        }
        run_path = OUT_DIR / f"local-{label}-{domain}.json"
        run_path.write_text(json.dumps(run_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        report = score_domain(domain, responses, traditions)
        report["model"] = label
        if args.with_gate:
            report["gateFailures"] = gate_failures
        report_path = OUT_DIR / f"local-{label}-{domain}.report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{domain}: {report['passed']}/{report['total']} ({report['score_pct']}%)")
        summary.append(report)

    all_passed = sum(r["passed"] for r in summary)
    all_total = sum(r["total"] for r in summary)
    print(f"TOTAL: {all_passed}/{all_total} ({round(100 * all_passed / all_total, 1) if all_total else 0}%)")
    return 0 if all_passed == all_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Evaluate a local MLX model/adapter on Sophia domain benchmarks.

This is the Mac/Apple Silicon counterpart to ``tools/eval_local_model.py`` for
MLX-LM adapters produced by ``mlx_lm lora``. It writes the same
``benchmark/model_runs/*.report.json`` shape so ``tools/eval_ladder.py`` can
summarize base/base+gate/adapter/adapter+gate.
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

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_domain_channels  # noqa: E402
from agent.gate import check_response  # noqa: E402

OUT_DIR = ROOT / "benchmark" / "model_runs"
SYSTEM = (
    "You are a Sophia AGI instructor using source discipline across philosophy, psychology, "
    "history, and religion. Name authors precisely, deny lineage-merge traps, label myths, "
    "state uncertainty when needed, and end with a concise 中文 summary."
)


def slug(name: str) -> str:
    return name.replace("/", "-").replace(" ", "-").lower()


def load_benchmarks() -> dict[str, list[dict]]:
    return {domain: load_json(path).get("cases", []) for domain, path in DOMAIN_BENCH.items()}


def _prompt(tokenizer: Any, question: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "apply_chat_template"):
        return tokenizer.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"<|system|>\n{SYSTEM}\n<|user|>\n{question}\n<|assistant|>\n"


def generate_answer(model: Any, tokenizer: Any, question: str, *, max_tokens: int) -> str:
    from mlx_lm import generate

    return generate(model, tokenizer, _prompt(tokenizer, question), max_tokens=max_tokens, verbose=False).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter", default=None, help="MLX adapter directory")
    parser.add_argument("--domains", nargs="*", default=list(DOMAIN_BENCH.keys()))
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--with-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    benches = load_benchmarks()
    total_cases = sum(len(benches[d]) for d in args.domains if d in benches)
    label = slug(Path(args.adapter).name if args.adapter else args.model)
    print(f"MLX model: {args.model}" + (f" + adapter {args.adapter}" if args.adapter else ""))
    print(f"Domains: {', '.join(args.domains)} | cases: {total_cases}")
    if args.dry_run:
        return 0

    try:
        from mlx_lm import load
    except ImportError:
        print("Install: pip install mlx-lm")
        return 1

    try:
        model, tokenizer = load(args.model, adapter_path=args.adapter)
    except Exception as exc:
        print(f"MLX load failed: {type(exc).__name__}: {exc}")
        return 1

    traditions = load_json(ROOT / "data" / "traditions.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for domain in args.domains:
        responses: dict[str, str] = {}
        gate_failures = 0
        for case in benches.get(domain, []):
            answer = generate_answer(model, tokenizer, case["question"], max_tokens=args.max_tokens)
            responses[case["id"]] = answer
            if args.with_gate:
                gate = check_response(answer, mode="advisor", question=case["question"], strict_attribution=True)
                if not gate.get("passed", True):
                    gate_failures += 1
            print(f"  {domain}/{case['id']}...")
        run_payload = {"model": label, "domain": domain, "date": datetime.now(timezone.utc).isoformat(), "responses": responses}
        (OUT_DIR / f"local-{label}-{domain}.json").write_text(json.dumps(run_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report = score_domain_channels(domain, responses, traditions)
        report["model"] = label
        report["backend"] = "mlx"
        if args.with_gate:
            report["gateFailures"] = gate_failures
        (OUT_DIR / f"local-{label}-{domain}.report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{domain}: {report['passed']}/{report['total']} ({report['score_pct']}%)")
        summary.append(report)
    all_passed = sum(r["passed"] for r in summary)
    all_total = sum(r["total"] for r in summary)
    print(f"TOTAL: {all_passed}/{all_total} ({round(100 * all_passed / all_total, 1) if all_total else 0}%)")
    return 0 if all_passed == all_total else 1


if __name__ == "__main__":
    raise SystemExit(main())

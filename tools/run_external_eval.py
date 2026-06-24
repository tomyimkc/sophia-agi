#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run an external-oracle eval: score a model against GOLD answers (not the gate).

Dataset-agnostic. A small style-sample ships so it runs offline; point --dataset
at the real GSM8K / GAIA / ARC JSONL ({question, answer}) to run the real thing.

    python tools/run_external_eval.py                                   # offline plumbing (mock)
    python tools/run_external_eval.py --model ollama:llama3.2:3b
    python tools/run_external_eval.py --dataset path/to/gsm8k.jsonl --model deepseek

HONEST SCOPE: the committed sample is a 10-item style demo, NOT a benchmark
result. A real external-eval number requires the actual public dataset and is
only quotable with N, the model, and the dataset version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import external_eval  # noqa: E402

SAMPLE = ROOT / "eval" / "external" / "gsm8k-style-sample.jsonl"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _mock_solver(item: dict) -> str:
    # offline plumbing only: returns the gold (proves the harness, not the model)
    return str(item["answer"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=str(SAMPLE))
    ap.add_argument("--model", default=None, help="model spec; omit for offline mock plumbing")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scorer", choices=["numeric", "symbolic"], default="numeric",
                    help="numeric exact-match (GSM8K-style) or symbolic equivalence (MATH-style, sympy)")
    args = ap.parse_args(argv)

    items = _load(Path(args.dataset))
    if args.limit:
        items = items[: args.limit]

    symbolic = args.scorer == "symbolic"
    scorer = external_eval.score_item_symbolic if symbolic else None
    if args.model:
        from agent.model import default_client

        client = default_client(args.model)
        sys_prompt = ("Solve the problem. Show brief reasoning and put the final answer in \\boxed{}."
                      if symbolic else
                      "Solve the problem. Show brief reasoning and end with the final number only.")
        solver = lambda it: getattr(client.generate(sys_prompt, it["question"]), "text", "") or ""
        label = args.model
    else:
        solver, label = _mock_solver, "mock (plumbing only)"

    report = external_eval.run_dataset(items, solver, scorer=scorer)
    is_sample = Path(args.dataset).resolve() == SAMPLE.resolve()
    print(f"dataset: {Path(args.dataset).name}{'  [STYLE SAMPLE — not a benchmark result]' if is_sample else ''}")
    print(f"model:   {label}")
    print(f"accuracy: {report['accuracy']:.1%}  ({report['correct']}/{report['n']})")
    print(f"oracle:  {report['oracle']}")
    if args.model and is_sample:
        print("\nNOTE: point --dataset at the real GSM8K/GAIA/ARC JSONL for a citable external number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

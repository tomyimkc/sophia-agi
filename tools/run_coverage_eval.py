#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the decomposition/coverage benchmark live (OpenRouter, no GPU).

solo vs swarm (complementary facet-agents + merge) aspect coverage on the same subject model;
aspect coverage judged by an independent LLM (judge != subject); paired bootstrap CI.

  OPENROUTER_API_KEY=... python tools/run_coverage_eval.py \
      --subject meta-llama/llama-3.2-3b-instruct --judge deepseek/deepseek-chat --n 15
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import swarm_coverage_eval as cov  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"


def _call(model: str, key: str, system: str, user: str, max_tokens: int) -> str:
    payload = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                          "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
    for _ in range(3):
        p = subprocess.run(["curl", "-sS", "--max-time", "90", API,
                            "-H", f"Authorization: Bearer {key}", "-H", "content-type: application/json",
                            "-H", "HTTP-Referer: https://github.com/tomyimkc/sophia-agi", "-H", "X-Title: Sophia",
                            "--data", payload], capture_output=True, text=True)
        try:
            return json.loads(p.stdout)["choices"][0]["message"]["content"].strip()
        except Exception:
            continue
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", default="meta-llama/llama-3.2-3b-instruct")
    ap.add_argument("--judge", default="deepseek/deepseek-chat")
    ap.add_argument("--pack", type=Path, default=ROOT / "data" / "coverage" / "pack_v1.jsonl")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        print("OPENROUTER_API_KEY required.", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in args.pack.read_text().splitlines() if l.strip()][: args.n]
    tasks = [cov.CoverageTask(r["question"], tuple(r["aspects"])) for r in rows]
    print(f"subject={args.subject} judge={args.judge} questions={len(tasks)} "
          f"aspects={sum(len(t.aspects) for t in tasks)} …", flush=True)

    model_fn = lambda system, user: _call(args.subject, key, system, user, 512)

    def judge_fn(answer: str, aspect: str) -> int:
        if not answer.strip():
            return 0
        sys_p = ("Does the ANSWER meaningfully address the given ASPECT of the question? "
                 "Answer one word: YES or NO.")
        out = _call(args.judge, key, sys_p, f'ASPECT: "{aspect}"\n\nANSWER: "{answer[:1500]}"\n\nCovered?', 4).upper()
        return 1 if "YES" in out else 0

    rep = cov.run_coverage(tasks, model_fn, judge_fn, subject=args.subject)
    out = rep.to_dict()
    print(json.dumps(out, indent=2))
    if args.out:
        args.out.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

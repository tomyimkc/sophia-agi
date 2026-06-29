#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DIAGNOSTIC: score a live OpenAI-compatible endpoint on the EXACT held-out
math-step split the registered RLVR eval uses (--task step --step-domain math),
swapping ONLY the generation source.

Methodological identity with tools/eval_rlvr_adapter.py::run_eval_step:
  * held-out split   -> provenance_bench.math_dataset.build_math_rl_dataset()
                        (FIXED explicit `split` field -> eval set is the same
                         across seeds/eval_frac; 60 problems, 3 families)
  * prompt wrap      -> provenance_bench.step_reward.STEP_INSTRUCTION + prompt
  * per-step oracle  -> provenance_bench.step_reward.reward_for_completion
                        (via the repo's _score_step, imported, NOT reimplemented)
  * metrics          -> eval_rlvr_adapter._score_step: passAt1 (verdict==accepted,
                        i.e. final correct AND every step machine-verified),
                        verifiedStepCoverage, meanReward, passAt1ByFamily

The ONLY thing changed vs the registered eval is WHERE completions come from:
here, an HTTP POST to {endpoint}/chat/completions instead of a local HF/PEFT
generate(). This is a single-model (base) reading, not a base-vs-adapter delta.

This is a DIAGNOSTIC: it never sets canClaimAGI and never tunes for a number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path("/home/tomyimkc/sophia-agi")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the repo's oracle + metrics + split loader (do NOT reimplement).
from provenance_bench import math_dataset, step_reward  # noqa: E402
from tools.eval_rlvr_adapter import _score_step  # noqa: E402


def generate(endpoint: str, model: str, prompt: str, *, max_tokens: int,
             api_key: str = "none", retries: int = 5, timeout: int = 600) -> str:
    """POST one chat completion (OpenAI schema, temperature 0) with backoff."""
    url = endpoint.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }).encode("utf-8")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            return payload["choices"][0]["message"]["content"] or ""
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                json.JSONDecodeError, TimeoutError) as exc:
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"generation failed after {retries} tries: {last!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL (.../v1)")
    ap.add_argument("--model", required=True, help="model id served at the endpoint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=0, help="debug subset size (0 = full split)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--api-key", default="none")
    args = ap.parse_args(argv)

    domain = "math"
    # EXACT registered held-out split (fixed `split` field => seed/eval_frac inert).
    data = math_dataset.build_math_rl_dataset(eval_frac=args.eval_frac, seed=args.seed)
    problems = data["eval_problems"]
    if data["family_intersection"]:
        raise SystemExit(f"contaminated split: {data['family_intersection']}")
    if args.limit:
        problems = problems[: args.limit]

    completions: dict[str, str] = {}
    for i, p in enumerate(problems, 1):
        wrapped = step_reward.STEP_INSTRUCTION + p["prompt"]  # identical wrap
        text = generate(args.endpoint, args.model, wrapped, max_tokens=args.max_tokens,
                        api_key=args.api_key)
        completions[p["id"]] = text
        print(f"[gen] {i}/{len(problems)} {p['id']}", flush=True)

    # EXACT registered metrics + oracle (step_reward via _score_step).
    score = _score_step(problems, completions, domain)

    report = {
        "benchmark": "rlvr-adapter-heldout",
        "task": "step",
        "stepDomain": domain,
        "mode": "live-endpoint-diagnostic",
        "diagnostic": True,
        "registeredResult": False,
        "canClaimAGI": False,
        "endpoint": args.endpoint,
        "model": args.model,
        "heldoutFile": str(math_dataset.DATA),
        "verifier": "provenance_bench.step_reward.reward_for_completion (agent.step_verifier)",
        "metricsFrom": "tools.eval_rlvr_adapter._score_step",
        "split": {
            "evalProblems": len(problems),
            "fullSplitN": len(data["eval_problems"]),
            "seed": args.seed,
            "evalFrac": args.eval_frac,
            "evalFamilies": sorted({p["family"] for p in problems}),
            "evalSealed": data["eval_sealed"],
            "familyIntersection": data["family_intersection"],
        },
        "metrics": {k: v for k, v in score.items() if k != "rows"},
        "rows": score["rows"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    m = report["metrics"]
    print(f"wrote {args.out}")
    print(f"SUMMARY model={args.model} n={m['n']} passAt1={m['passAt1']} "
          f"VSC={m['verifiedStepCoverage']} meanReward={m['meanReward']} "
          f"byFamily={m['passAt1ByFamily']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

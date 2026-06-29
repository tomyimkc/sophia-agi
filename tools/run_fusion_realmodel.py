#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Real-model REFLEX FUSION measurement (OpenAI-compatible providers).

The real-model version of ``reasoning/instinct_fusion``: a real model produces N sampled
abstain-sets per belief-revision case, and we compute BOTH detectors on those real answers —
A = self-consistency disagreement (label-free) and B = the real ``okf`` grounding-closure
violation — then their fused d′ against the break-even bar.

Security & cost (read before running):
  - Key is read ONLY from the env var named by ``--key-env`` (default per provider). Never
    hard-coded, logged, or written to any artifact. ``export DEEPSEEK_API_KEY=...`` etc.
  - Spends real credits; sends prompts to an external service. Prints the call budget and
    requires ``--yes``. Fails LOUD on any API error (never folds a failure into the data).
  - One model = one judge family at one seed → ``candidateOnly``; not promotable. The
    no-overclaim gate still needs ≥2 families, ≥3 seeds, CI.

Provider presets (both verified OpenAI-compatible /v1/chat/completions):
  deepseek : https://api.deepseek.com         key DEEPSEEK_API_KEY  model deepseek-chat
  llmhub   : https://api.llmhub.com.cn/v1     key LLMHUB_API_KEY    model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reasoning.instinct_reflex_eval import auc, d_prime, load_cases  # noqa: E402
from reasoning.instinct_fusion import (  # noqa: E402
    _majority, _reflex_A, _reflex_B, _reflex_B2, _true_abstain, breakeven_snr, fuse,
)
from tools.run_reflex_openrouter import _build_prompt, _call_model  # noqa: E402

PRESETS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "key_env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
    "llmhub": {"base_url": "https://api.llmhub.com.cn/v1", "key_env": "LLMHUB_API_KEY", "model": "claude-haiku-4-5-20251001"},
}


def _parse_set(text: str) -> frozenset[str]:
    """Parse a model completion into the proposed abstain SET (ids)."""
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[-1]
    start, end = t.find("["), t.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(t[start:end + 1])
            return frozenset(str(x) for x in arr)
        except (json.JSONDecodeError, TypeError):
            pass
    return frozenset({f"UNPARSEABLE::{t[:30]}"})


def run(model: str, base_url: str, key: str, n_samples: int, limit: int | None,
        temperature: float, timeout: float, out: Path | None) -> int:
    cases = load_cases()
    if limit:
        cases = cases[:limit]
    bar = breakeven_snr()

    a_scores: list[float] = []
    b_scores: list[float] = []
    b2_scores: list[float] = []
    labels: list[bool] = []
    per_case = []
    for i, case in enumerate(cases):
        true_set = _true_abstain(case)
        removed = set(case.get("remove", []))
        true_eff = frozenset(true_set - removed)
        samples = []
        for _ in range(n_samples):
            # No try/except: a ModelCallError propagates and aborts (failures are not data).
            raw = _call_model(_build_prompt(case), model=model, base_url=base_url,
                              key=key, temperature=temperature, timeout=timeout)
            samples.append(_parse_set(raw))
        majority = _majority(samples)
        majority_eff = frozenset(majority - removed)
        is_error = (majority_eff != true_eff)
        a = _reflex_A(samples)
        b = _reflex_B(majority, true_set, removed)    # over-abstention
        b2 = _reflex_B2(majority, true_set, removed)  # under-abstention (completeness)
        a_scores.append(a); b_scores.append(b); b2_scores.append(b2); labels.append(is_error)
        per_case.append({"id": case.get("id"), "is_error": is_error,
                         "A": round(a, 4), "B": round(b, 4), "B2": round(b2, 4),
                         "samples": [sorted(s) for s in samples]})  # raw sets ⇒ free re-scoring
        print(f"  [{i + 1}/{len(cases)}] {case.get('id')}: "
              f"{'ERR ' if is_error else 'ok  '} A={a:.3f} B={b:.3f} B2={b2:.3f}", file=sys.stderr)

    def split(scores):
        return ([s for s, e in zip(scores, labels) if e],
                [s for s, e in zip(scores, labels) if not e])

    def dp(scores):
        e, c = split(scores)
        return d_prime(e, c), auc(e, c)

    dpa, auca = dp(a_scores); dpb, aucb = dp(b_scores); dpb2, aucb2 = dp(b2_scores)
    detectors = {"A": a_scores, "B": b_scores, "B2": b2_scores}
    dprimes = {"A": dpa, "B": dpb, "B2": dpb2}
    # Equal-weight 3-detector fusion vs quality-weighted (weight = max(0, d′), Fisher-style).
    fused_eq = fuse(detectors, {k: 1.0 for k in detectors})
    qw = {k: max(0.0, v) if math.isfinite(v) else 0.0 for k, v in dprimes.items()}
    fused_qw = fuse(detectors, qw)
    dpfe, aucfe = dp(fused_eq); dpfq, aucfq = dp(fused_qw)
    n = len(cases)

    def r(x):
        return round(x, 4) if isinstance(x, float) and math.isfinite(x) else x

    report = {
        "schema": "sophia.reasoning.fusion.realmodel.v2",
        "model": model, "base_url": base_url, "n_cases": n,
        "n_samples": n_samples, "temperature": temperature,
        "base_error": round(sum(labels) / n, 4) if n else 0.0,
        "d_prime": {"A": r(dpa), "B": r(dpb), "B2": r(dpb2),
                    "fused_equal": r(dpfe), "fused_qualityweighted": r(dpfq)},
        "auc": {"A": r(auca), "B": r(aucb), "B2": r(aucb2),
                "fused_equal": r(aucfe), "fused_qualityweighted": r(aucfq)},
        "qw_weights": {k: r(v) for k, v in qw.items()},
        "breakeven_snr": bar,
        "clears": {k: bool(math.isfinite(v) and v >= bar)
                   for k, v in {"A": dpa, "B": dpb, "B2": dpb2,
                                "fused_equal": dpfe, "fused_qualityweighted": dpfq}.items()},
        "candidateOnly": True, "level3Evidence": False,
        "boundary": "single model = 1 family @ 1 seed; qw weights fit in-sample (cross-val is a "
                    "follow-up). Not promotable; no-overclaim gate needs >=2 families, >=3 seeds, CI.",
    }
    print(json.dumps(report, indent=2))
    if out:
        out.write_text(json.dumps({"report": report, "per_case": per_case}, indent=2))
        print(f"\nwrote {out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=sorted(PRESETS), default="deepseek")
    p.add_argument("--model", default=None, help="override the preset model id")
    p.add_argument("--base-url", default=None, help="override the preset base url")
    p.add_argument("--key-env", default=None, help="override the env var holding the api key")
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--yes", action="store_true", help="confirm: this spends real credits")
    args = p.parse_args(argv)

    preset = PRESETS[args.provider]
    model = args.model or preset["model"]
    base_url = args.base_url or preset["base_url"]
    key_env = args.key_env or preset["key_env"]
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"No API key in env var {key_env}. export {key_env}=... and re-run.")

    n_cases = args.limit or len(load_cases())
    budget = n_cases * args.samples
    print(f"Plan: provider={args.provider} model={model} cases={n_cases} samples={args.samples} "
          f"=> {budget} API calls (real credits).", file=sys.stderr)
    if not args.yes:
        print("Refusing to spend without --yes.", file=sys.stderr)
        return 2
    return run(model, base_url, key, args.samples, args.limit, args.temperature, args.timeout, args.out)


if __name__ == "__main__":
    sys.exit(main())

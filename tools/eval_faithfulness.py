#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the held-out faithfulness eval — counterfactual_grounding_rate of a policy.

OFFLINE (CI, no network):
    python tools/eval_faithfulness.py --mock

LIVE BASELINE (no GPU — an off-the-shelf API model AS the policy, a DIFFERENT family as
the entailment verifier; measures how retrieval-faithful a stock model is, claims
NOTHING about a trained model):
    python tools/eval_faithfulness.py \
      --policy llmhub:gemini-2.5-flash-lite --entailment deepseek --limit 24

This is the instrument from provenance_bench/faithfulness_eval.py wired to live seams.
The result is labelled candidate/illustrative: single model, single entailment family,
N below the pre-registered requiredN (measurement_spec.json). canClaimAGI:false. No GO
verdict is emitted; this is a baseline + an end-to-end instrument check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import faithfulness_eval, faithfulness_seams  # noqa: E402

OUT_DIR = ROOT / "agi-proof" / "benchmark-results" / "faithfulness"


def _make_policy(spec: str):
    """spec = '<provider>:<model>' (provider in {llmhub, deepseek}). Returns a
    generate(query, context_chunks) -> str seam that answers ONLY from the sources."""
    from provenance_bench.faithfulness_grpo import _build_prompt

    provider, _, model = spec.partition(":")
    key_file = f"private/secrets/{provider}_api_key"
    if provider == "llmhub":
        from agent import llmhub_llm as cli
        model = model or "gemini-2.5-flash-lite"
    elif provider == "deepseek":
        from agent import deepseek_llm as cli
        model = model or cli.DEFAULT_MODEL
    else:
        raise SystemExit(f"unknown policy provider {provider!r} (use llmhub: or deepseek:)")

    def generate(query: str, context_chunks: list) -> str:
        prompt = _build_prompt(query, context_chunks)
        try:
            return cli.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model, api_key_file=key_file, max_tokens=160, temperature=0.0,
            ).strip()
        except Exception as exc:  # noqa: BLE001 — a failed gen becomes an empty answer
            print(f"  policy gen failed ({type(exc).__name__}); treating as no-answer")
            return ""
    return generate, f"{provider}:{model}"


def _build_entailment(provider: str, model: str | None):
    """Entailment verifier fn. 'lexical' = the deterministic offline placeholder; a
    provider name = the live LLM verifier (keys from private/secrets/ or pod env)."""
    if provider == "lexical":
        return faithfulness_seams.lexical_entailment
    return faithfulness_seams.entailment_from_provider(provider, model=model)


def _seams(args, entailment):
    return dict(
        retrieve=faithfulness_seams.make_ai_search_retrieve(top_k=args.top_k),
        extract_claims=faithfulness_seams.heuristic_extract_claims,
        verify_claim=faithfulness_seams.make_entailment_verify(entailment),
        check_correct=lambda a, g: bool(g) and str(g).lower() in a.lower(),
    )


def _run_compare(args) -> int:
    """Base-vs-adapter faithfulness contrast on a local HF policy with a trained LoRA."""
    from provenance_bench.faithfulness_grpo import cases_from_rl_dataset

    provider, _, model = args.policy.partition(":")
    if provider != "hf" or not args.adapter:
        raise SystemExit("--compare needs --policy hf:<model> and --adapter <path>")

    base_gen, adapter_gen, label = faithfulness_eval.make_hf_compare_policies(model, args.adapter)
    seams = _seams(args, _build_entailment(args.entailment, args.entailment_model))
    cases = cases_from_rl_dataset(seed=args.seed, limit=args.limit, split="eval")
    print(f"compare policy={label}  entailment={args.entailment}  cases={len(cases)}")

    c = faithfulness_eval.compare(cases, base_generate=base_gen,
                                  adapter_generate=adapter_gen, **seams)
    report = {
        "benchmark": "reasoning-core-faithfulness-compare",
        "policy": label,
        "adapter": str(args.adapter),
        "entailmentVerifier": args.entailment,
        "nCases": len(cases),
        "baseGroundingRate": c["baseRate"],
        "adapterGroundingRate": c["adapterRate"],
        "meanDiff": c["meanDiff"],
        "pairedBootstrapCI95": c["pairedBootstrapCI95"],
        "nPaired": c["nPaired"],
        "caveats": [
            "candidate — not a GO verdict; the GO/NO-GO is tools/claim_gate.py against the "
            + "pre-registered measurement_spec.json (powered N, >=2 judge families, decontam).",
            f"N below pre-registered requiredN 377 unless nPaired is large; nPaired={c['nPaired']}.",
            "single entailment family; the >=2-family judge panel is a separate construct.",
        ],
        "claimStatus": "candidate — base-vs-adapter faithfulness contrast on the trained "
                       "adapter; promotion requires a claim_gate GO on the full spec. canClaimAGI:false.",
        "claimCeiling": "candidate_only; canClaimAGI:false; narrow corpus-bound feasibility",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "baseGroundingRate", "adapterGroundingRate", "meanDiff", "pairedBootstrapCI95",
        "nPaired")}, indent=2))
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="offline instrument invariants (CI)")
    ap.add_argument("--policy", default="llmhub:gemini-2.5-flash-lite",
                    help="policy '<provider>:<model>' (llmhub:/deepseek:) or 'hf:<model>' "
                         "(local HF model; with --compare + --adapter does base-vs-adapter)")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter path for an hf: policy (enables --compare)")
    ap.add_argument("--compare", action="store_true",
                    help="base-vs-adapter contrast (requires --policy hf:<model> --adapter <path>)")
    ap.add_argument("--entailment", default="deepseek", choices=["deepseek", "llmhub", "lexical"],
                    help="entailment verifier (provider, or 'lexical' offline placeholder)")
    ap.add_argument("--entailment-model", default=None)
    ap.add_argument("--limit", type=int, default=24, help="number of held-out cases")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "baseline-eval.json")
    args = ap.parse_args(argv)

    if args.mock:
        ok, detail = faithfulness_eval.offline_invariants()
        print(json.dumps(detail["checks"], indent=2))
        print("FAITHFULNESS EVAL INSTRUMENT OK ✓" if ok else "INSTRUMENT INVARIANTS NOT MET ✗")
        return 0 if ok else 1

    if args.compare:
        return _run_compare(args)

    # --- live baseline ---
    from provenance_bench.faithfulness_grpo import cases_from_rl_dataset

    generate, policy_id = _make_policy(args.policy)
    seams = _seams(args, _build_entailment(args.entailment, args.entailment_model))
    cases = cases_from_rl_dataset(seed=args.seed, limit=args.limit, split="eval")
    print(f"policy={policy_id}  entailment={args.entailment}  cases={len(cases)}  "
          f"(judge != subject: {args.entailment not in policy_id})")

    result = faithfulness_eval.evaluate(cases, generate=generate, **seams)
    agg = result["aggregate"]
    coverage = (agg["casesWithClaims"] / len(cases)) if cases else None
    underpowered = agg["mdeAtN"] is None or agg["mdeAtN"] > 0.10  # vs pre-registered mde
    report = {
        "benchmark": "reasoning-core-faithfulness-baseline",
        "policy": policy_id,
        "entailmentVerifier": f"{args.entailment}:{args.entailment_model or 'default'}",
        "judgeNeqSubject": args.entailment not in policy_id,
        "nCases": len(cases),
        "counterfactualGroundingRate": agg["rate"],
        "fixedNCI95": agg["fixedNCI95"],
        "anytimeValidCS95": agg["anytimeValidCS95"],
        "mdeAtN": agg["mdeAtN"],
        "knowledgeClaims": agg["knowledgeClaims"],
        "groundedClaims": agg["groundedClaims"],
        "casesWithClaims": agg["casesWithClaims"],
        "extractorCoverage": coverage,
        "abstentionRate": agg["abstentionRate"],
        "underpowered": underpowered,
        "caveats": [
            "UNTRAINED off-the-shelf baseline, not an uplift — no base-vs-adapter contrast, no GO verdict.",
            f"underpowered: {agg['knowledgeClaims']} knowledge claims vs requiredN 377 "
            f"(mde@N={agg['mdeAtN']}); the CI is wide and the point estimate is illustrative.",
            f"heuristic-extractor coverage {coverage}: only attribution-shaped claims are "
            "measured, which biases WHICH claims enter the rate.",
            "single policy model + single entailment family; the >=2-family judge panel "
            + "(measurement_spec.json) is separate and unrun.",
        ],
        "claimStatus": "candidate/illustrative — single model, single entailment family, "
                       "N below pre-registered requiredN (see measurement_spec.json). This is "
                       "an UNTRAINED baseline, not an uplift; no GO verdict. canClaimAGI:false.",
        "claimCeiling": "candidate_only; canClaimAGI:false; narrow corpus-bound feasibility",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "counterfactualGroundingRate", "fixedNCI95", "knowledgeClaims", "groundedClaims",
        "casesWithClaims", "abstentionRate")}, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

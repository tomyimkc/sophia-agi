#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Multi-judge (cross-family) corroboration of the fact-check ClaimReview oracle.

Polls several LLM judge families for the entailment relation of each Google
ClaimReview source, takes their fail-closed consensus as the gate's Layer-2
entailment, and reports inter-judge Cohen's kappa.

Judges are model specs: ``llmhub:<model>`` (cross-provider gateway,
``LLMHUB_API_KEY``) or ``deepseek:<model>`` (direct, ``DEEPSEEK_API_KEY``).

    GOOGLE_FACTCHECK_API_KEY=... LLMHUB_API_KEY=... DEEPSEEK_API_KEY=... \
      python tools/run_factcheck_multijudge.py \
        --pack eval/fact_check/google_factcheck_v2.jsonl \
        --judge llmhub:claude-opus-4-8 --judge llmhub:gpt-4o-mini --judge deepseek:deepseek-v4-flash \
        --out agi-proof/fact-check-live/<name>.json

PROVENANCE CAVEAT: judges reached through a single gateway share infrastructure;
model identity is not cryptographically verified. This is cross-family-via-proxy
corroboration, weaker than direct first-party keys. Not Level-3 evidence;
``canClaimAGI`` stays false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.fact_check_eval import load_jsonl, run_fact_check_eval, write_report  # noqa: E402
from agent.factcheck_nli import MultiJudgeNLI, NLIEntailment  # noqa: E402
from agent.factcheck_oracle import GoogleFactCheckOracle  # noqa: E402
from provenance_bench.consensus import cohen_kappa  # noqa: E402

_REL_TO_INT = {"irrelevant": 0, "entails": 1, "contradicts": 2}


def _complete_for(spec: str):
    provider, _, model = spec.partition(":")
    if not model:
        raise SystemExit(f"judge spec must be provider:model, got {spec!r}")
    if provider == "llmhub":
        from agent.llmhub_llm import make_complete
        return make_complete(model=model, temperature=0.0, max_tokens=24)
    if provider == "deepseek":
        from agent.deepseek_llm import make_complete
        return make_complete(model=model, temperature=0.0, max_tokens=24)
    raise SystemExit(f"unknown judge provider {provider!r} (use llmhub: or deepseek:)")


def _kappa_matrix(record: dict[str, dict[str, str]], judges: list[str]) -> dict[str, float | None]:
    source_ids = sorted(record)
    seqs = {j: [_REL_TO_INT[record[sid][j]] for sid in source_ids] for j in judges}
    out: dict[str, float | None] = {}
    for i, a in enumerate(judges):
        for b in judges[i + 1:]:
            out[f"{a}_vs_{b}"] = cohen_kappa(seqs[a], seqs[b])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default=str(ROOT / "eval" / "fact_check" / "google_factcheck_v2.jsonl"))
    ap.add_argument("--judge", action="append", dest="judges", default=None,
                    help="provider:model judge spec (repeatable; >=2 for kappa)")
    ap.add_argument("--google-page-size", type=int, default=5)
    ap.add_argument("--min-agree", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-fabrication-rate", type=float, default=0.05)
    args = ap.parse_args(argv)

    judges = args.judges or ["llmhub:claude-opus-4-8", "llmhub:gpt-4o-mini", "deepseek:deepseek-v4-flash"]
    oracle = GoogleFactCheckOracle(page_size=args.google_page_size)
    if not oracle.enabled:
        print("ERROR: GOOGLE_FACTCHECK_API_KEY is required", file=sys.stderr)
        return 2

    judge_objs = {spec: NLIEntailment(complete=_complete_for(spec), source_types=None) for spec in judges}
    record: dict[str, dict[str, str]] = {}
    multijudge = MultiJudgeNLI(judge_objs, min_agree=args.min_agree, source_types={"factcheck"}, record=record)

    rows = load_jsonl(args.pack)
    report = run_fact_check_eval(
        rows, retriever=oracle.retriever, entailment=multijudge,
        live_backend=True, target_fabrication_rate=args.target_fabrication_rate,
    )

    kappa = _kappa_matrix(record, judges) if len(judges) >= 2 else {}
    vals = [k for k in kappa.values() if k is not None]
    report["multiJudge"] = {
        "judges": judges,
        "minAgree": args.min_agree,
        "sourcesJudged": len(record),
        "interJudgeKappa": kappa,
        "meanPairwiseKappa": round(sum(vals) / len(vals), 4) if vals else None,
        "provenanceCaveat": (
            "Judges reached via the llmhub gateway share infrastructure; model identity not "
            "cryptographically verified. Cross-family-via-proxy corroboration, weaker than direct "
            "first-party keys. Not Level-3; _is_validated NOT auto-cleared by this artifact."
        ),
        "canClaimAGI": False,
    }
    write_report(report, args.out)
    print(json.dumps({
        "out": args.out, "n": report["n"], "judges": judges,
        "metrics": report["metrics"], "multiJudge": report["multiJudge"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

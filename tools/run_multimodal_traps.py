#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the image-grounded hallucination-trap suite and print a no-overclaim report.

Offline by default (deterministic mock VLMs, no network/GPU):

    python tools/run_multimodal_traps.py --answer mock:grounded   --runs 3
    python tools/run_multimodal_traps.py --answer mock:credulous  --runs 3

Real VLM (opt-in; needs OPENAI_API_KEY + Pillow, sends rendered PNGs):

    python tools/run_multimodal_traps.py --answer openai:gpt-4o --runs 5 \
        --judge-spec anthropic:claude-... --judge-spec deepseek:deepseek-chat

A headline number is only ``validated`` when it clears every no-overclaim check:
a real (non-mock) model, >=2 judges from distinct provider families with Cohen's
kappa >= 0.40, >=3 runs, and a computed CI. Mock runs are always illustrative.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multimodal_bench import runner  # noqa: E402
from multimodal_bench.model import resolve_answer_fn  # noqa: E402


def _build_judge(judge_specs: "list[str] | None"):
    """Lexical judge by default; a real multi-family consensus judge if specs given."""
    if not judge_specs:
        return None, None
    if len(judge_specs) < 2:
        sys.exit("error: a consensus judge needs >=2 --judge-spec values (distinct families)")
    from agent import model as agent_model  # late import: only for real runs
    from multimodal_bench import judge as judge_mod

    system = ("You are an independent referee scoring a vision model's answer "
              "against a known ground-truth. Reply with exactly one word.")

    def make(spec):
        def fn(answer, trap):
            user = (
                f"Question asked of the vision model: {trap['question']!r}\n"
                f"Ground-truth answer: {trap['gold_answer']!r}\n"
                f"Vision model replied: {answer!r}\n"
                "Reply GOLD (it matched the ground truth), TRAP (it gave a wrong "
                "answer), or ABSTAIN (it declined)."
            )
            verdict = (agent_model.complete(system, user, spec=spec) or "").strip().upper()
            return judge_mod.VisualJudgment(
                abstained="ABSTAIN" in verdict,
                hallucinated="TRAP" in verdict and "GOLD" not in verdict,
                affirmed_gold="GOLD" in verdict,
            )
        return fn

    consensus = judge_mod.make_consensus_judge([make(s) for s in judge_specs], labels=judge_specs)
    return consensus, judge_specs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answer", default="mock:grounded",
                    help="answer backend: mock:grounded|credulous|abstainer, or openai:<model>")
    ap.add_argument("--runs", type=int, default=3, help="number of repeated runs to pool")
    ap.add_argument("--judge-spec", action="append", dest="judge_specs",
                    help="LLM judge spec (repeat >=2 for a real consensus judge)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-synth", action="store_true",
                    help="also include the verifier-checked chart/table/document traps")
    ap.add_argument("--categories",
                    help="comma-separated category allowlist (scope the eval to these only)")
    ap.add_argument("--physical", action="store_true",
                    help="scope to the physical/2.5D axes (depth, occlusion, size, distance)")
    ap.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = ap.parse_args(argv)

    traps = runner.load_all_traps() if args.include_synth else runner.load_traps()
    if args.physical:
        traps = runner.filter_by_category(traps, runner.PHYSICAL_CATEGORIES)
    elif args.categories:
        traps = runner.filter_by_category(traps, [c.strip() for c in args.categories.split(",") if c.strip()])
    if not traps:
        sys.exit("error: no traps match the requested categories")
    answer_fn = resolve_answer_fn(args.answer)
    judge_fn, judge_specs = _build_judge(args.judge_specs)

    runs = [runner.run_cases(traps, answer_fn, judge_fn) for _ in range(args.runs)]
    report = runner.aggregate_runs(runs, seed=args.seed, model_spec=args.answer, judges=judge_specs)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"\nVisual hallucination-trap suite — {len(traps)} traps x {args.runs} runs ({args.answer})")
    print(f"  hallucination rate : {report['hallucinationRate']:.3f}  95% CI {report['ciHallucination']}")
    print(f"  grounding rate     : {report['groundingRate']:.3f}  95% CI {report['ciGrounding']}")
    print(f"  abstention rate    : {report['abstentionRate']:.3f}")
    print("  by category:")
    for cat, m in report["byCategory"].items():
        print(f"    {cat:<18} n={m['n']:<3} halluc={m['hallucinationRate']:.2f} ground={m['groundingRate']:.2f} abstain={m['abstentionRate']:.2f}")
    if report["judgeAgreement"]:
        ja = report["judgeAgreement"]
        print(f"  judge agreement    : {ja['meanPairwiseAgreement']} (kappa {ja['meanPairwiseKappa']}, {len(ja['judges'])} judges)")
    print(f"  VALIDATED (no-overclaim): {report['validated']}")
    for check, ok in report["validatedChecks"].items():
        print(f"    [{'x' if ok else ' '}] {check}")
    if not report["validated"]:
        print("  -> illustrative only; not a publishable headline. See RESULTS.md bar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

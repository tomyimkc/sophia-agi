#!/usr/bin/env python3
"""Run the Provenance Delta benchmark: build → run (alone vs gated) → score → report.

Examples
--------
    # offline smoke run (deterministic mock model, no API cost)
    python tools/run_provenance_delta.py --models mock

    # real headline run with an independent LLM-judge
    python tools/run_provenance_delta.py \
        --models anthropic,openai,grok,ollama:qwen2.5-7b \
        --llm-judge anthropic:claude-opus-4-8

The judge model MUST differ from the models under test (independence). Labels
are external; the gate is only the runtime treatment.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import aggregate, dataset, report, score  # noqa: E402
from provenance_bench.runner import run_cases  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "provenance-delta.public-report.json"
OUT_MD = ROOT / "agi-proof" / "benchmark-results" / "provenance-delta.md"


def _generator(spec: str):
    """Return a ``generate(system, user)`` callable for a model spec.

    ``mock`` uses the offline deterministic provider; any other spec resolves
    through the unified adapter (agent/model.py).
    """
    from agent.model import default_client

    client = default_client(spec)
    return lambda system, user: client.generate(system, user)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Provenance Delta benchmark runner")
    ap.add_argument("--models", default="mock", help="comma list of model specs (default: mock)")
    ap.add_argument("--on-fail", default="repair", help="guarded loop mode: repair|abstain|hedge|passthrough")
    ap.add_argument("--runs", type=int, default=1, help="runs per model; >1 enables bootstrap CIs")
    ap.add_argument("--llm-judge", default=None, help="model spec for an independent LLM-judge (else lexical screen)")
    ap.add_argument("--limit", type=int, default=0, help="cap number of cases (0 = all)")
    ap.add_argument("--emit-dataset", default=None, help="also write the case set as JSONL to this path")
    ap.add_argument("--out", default=str(OUT_JSON), help="report JSON path")
    args = ap.parse_args(argv)

    cases = dataset.build_cases()
    gate_records = dataset.build_gate_records()  # rules derived from cited misattributions
    if args.limit:
        cases = cases[: args.limit]
    if args.emit_dataset:
        n = dataset.write_jsonl(cases, Path(args.emit_dataset))
        print(f"wrote {n} cases -> {args.emit_dataset}")

    llm_judge_fn = None
    if args.llm_judge:
        from provenance_bench.llm_judge import make_llm_judge

        llm_judge_fn = make_llm_judge(args.llm_judge)

    per_model: dict = {}
    for spec in [s.strip() for s in args.models.split(",") if s.strip()]:
        print(f"running {spec} over {len(cases)} cases x{args.runs} run(s) ...")
        gen = _generator(spec)
        runs = [
            run_cases(cases, gen, on_fail=args.on_fail, records=gate_records, llm_judge_fn=llm_judge_fn)
            for _ in range(max(1, args.runs))
        ]
        scores = aggregate.aggregate_runs(runs) if args.runs > 1 else score.score(runs[0])
        per_model[spec] = {
            "scores": scores,
            "model": spec,
            "onFail": args.on_fail,
            "runs": args.runs,
            "judgeMethod": runs[0][0]["judge_method"] if runs and runs[0] else "lexical",
        }
        ci = f" CI{scores['ciDelta']}" if "ciDelta" in scores else ""
        print(
            f"  {spec}: halluc alone={scores['hallucinationRateAlone']:.1%} "
            f"gated={scores['hallucinationRateGated']:.1%} Δ={scores['delta']:.1%}{ci} "
            f"FP-cost={scores['falsePositiveCost']:.1%} coverage={scores['coverageRecall']:.1%}"
        )

    rpt = report.build_report(per_model, run_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    report.write_report(rpt, Path(args.out), OUT_MD)
    print(f"\nreport -> {args.out}\nmarkdown -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

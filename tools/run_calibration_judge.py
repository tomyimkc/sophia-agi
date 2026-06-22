#!/usr/bin/env python3
"""Corroborate the calibration scorer with one or more independent LLM judges.

Re-judges captured ablation responses (run_ablation_sophia --private-out dumps) with
each --judge model and compares, per answer, against the deterministic calibration
scorer. Reports each method's per-mode fabrication rate, the full Cohen's-κ matrix
(inter-judge agreement), and — with >=2 judges — a consensus fabrication stream.

    # single judge
    python tools/run_calibration_judge.py <pack.json> <private-*.json> --judge deepseek:deepseek-chat
    # two distinct families (headline-grade corroboration)
    python tools/run_calibration_judge.py <pack.json> <private-*.json> \
        --judge openai:gpt-4o --judge openai:claude-sonnet-4-6 --json

Offline: --judge mock uses the deterministic mock model. A caveat is printed when a
judge family matches the subject's.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench.calibration_judge import (  # noqa: E402
    cohen_kappa,
    consensus_fabricated,
    judge_answer,
    kappa_matrix,
)
from provenance_bench.calibration_score import score_answer  # noqa: E402

MODES = ("sophia-full", "raw-model", "raw-model-plus-tools")


def _judge_fn(spec: str):
    from agent.model import complete

    return lambda prompt: complete("You are a strict evaluator.", prompt, spec=spec, max_tokens=24)


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path)
    ap.add_argument("dumps", type=Path, nargs="+")
    ap.add_argument("--judge", action="append", dest="judges", default=None,
                    help="model spec for a judge (repeatable for multiple families)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    judges = args.judges or ["mock"]

    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    abstain = [c for c in pack["cases"] if c.get("epistemicLabel") == "abstain"]
    judge_fns = {spec: _judge_fn(spec) for spec in judges}

    # aligned per-answer 'fabricated?' streams, keyed by method
    streams: dict = {"scorer": [], **{j: [] for j in judges}}
    mode_rates: dict = {m: {src: [] for src in streams} for m in MODES}

    for dump_path in args.dumps:
        dump = json.loads(dump_path.read_text(encoding="utf-8"))
        for mode in MODES:
            responses = (dump.get(mode) or {}).get("responses", {})
            if not responses:
                continue
            run = {src: 0 for src in streams}
            n = 0
            for case in abstain:
                ans = responses.get(case["id"], "")
                n += 1
                sf = score_answer(ans, case)["fabricated"]
                streams["scorer"].append(sf); run["scorer"] += int(sf)
                for spec in judges:
                    fab = judge_answer(case.get("prompt", ""), ans, judge_fn=judge_fns[spec]) == "fabricated"
                    streams[spec].append(fab); run[spec] += int(fab)
            for src in streams:
                mode_rates[mode][src].append(round(run[src] / n, 4) if n else None)

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 4) if xs else None

    out = {
        "pack": str(args.pack), "judges": judges, "dumps": len(args.dumps),
        "nPairedAnswers": len(streams["scorer"]),
        "fabricationRate": {src: {m: _mean(mode_rates[m][src]) for m in MODES} for src in streams},
        "kappa": kappa_matrix(streams),
    }
    if len(judges) >= 2:
        cons = consensus_fabricated(*[streams[j] for j in judges])
        out["kappa"]["scorer_vs_consensus"] = cohen_kappa(streams["scorer"], cons)
        out["consensusFabricatedCount"] = sum(cons)
    out["caveats"] = [
        (f"judge '{j}' family may match the deepseek subject — method-level only"
         if "deepseek" in j.lower() else
         f"judge '{j}' is the offline mock model" if j == "mock" else
         f"judge '{j}' is a distinct family from the subject")
        for j in judges
    ]
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

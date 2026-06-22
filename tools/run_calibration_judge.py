#!/usr/bin/env python3
"""Corroborate the calibration scorer with an independent LLM judge.

Re-judges captured ablation responses (run_ablation_sophia --private-out dumps) with
an LLM and compares, per answer, against the deterministic calibration scorer. Reports
each method's per-mode fabrication rate and the Cohen's κ between them.

    python tools/run_calibration_judge.py <pack.json> <private-1.json> [<private-2.json> ...] \
        [--judge deepseek:deepseek-chat] [--json]

Offline: with --judge mock (or no key) the judge is the deterministic mock model.
Honest caveat printed when the judge family matches the subject's.
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

from provenance_bench.calibration_judge import cohen_kappa, judge_pack  # noqa: E402
from provenance_bench.calibration_score import score_answer  # noqa: E402

MODES = ("sophia-full", "raw-model", "raw-model-plus-tools")


def _judge_fn(spec: str):
    from agent.model import complete

    return lambda prompt: complete("You are a strict evaluator.", prompt, spec=spec, max_tokens=8)


def _abstain_cases(pack: dict) -> list:
    return [c for c in pack["cases"] if c.get("epistemicLabel") == "abstain"]


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", type=Path)
    ap.add_argument("dumps", type=Path, nargs="+", help="run_ablation_sophia --private-out JSON dump(s)")
    ap.add_argument("--judge", default="mock", help="model spec for the judge (e.g. deepseek:deepseek-chat)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    pack = json.loads(args.pack.read_text(encoding="utf-8"))
    abstain = _abstain_cases(pack)
    case_by_id = {c["id"]: c for c in abstain}
    judge_fn = _judge_fn(args.judge)

    judge_fab = {m: [] for m in MODES}      # per-run judge fabricationRate
    scorer_fab = {m: [] for m in MODES}     # per-run scorer fabricationRate
    paired_scorer: list = []                # per-answer fabricated? (scorer)
    paired_judge: list = []                 # per-answer fabricated? (judge)

    for dump_path in args.dumps:
        dump = json.loads(dump_path.read_text(encoding="utf-8"))
        for mode in MODES:
            responses = (dump.get(mode) or {}).get("responses", {})
            if not responses:
                continue
            jp = judge_pack(pack, responses, judge_fn=judge_fn)
            judge_fab[mode].append(jp["fabricationRate"])
            jlabel = {r["id"]: r["judge_label"] for r in jp["perCase"]}
            sfab = stotal = 0
            for cid, case in case_by_id.items():
                ans = responses.get(cid, "")
                s = score_answer(ans, case)
                stotal += 1
                sfab += int(s["fabricated"])
                paired_scorer.append(s["fabricated"])
                paired_judge.append(jlabel.get(cid) == "fabricated")
            scorer_fab[mode].append(round(sfab / stotal, 4) if stotal else None)

    def _mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 4) if xs else None

    out = {
        "pack": str(args.pack), "judge": args.judge, "dumps": len(args.dumps),
        "scorerFabricationRate": {m: _mean(scorer_fab[m]) for m in MODES},
        "judgeFabricationRate": {m: _mean(judge_fab[m]) for m in MODES},
        "kappa_scorer_vs_judge": cohen_kappa(paired_scorer, paired_judge),
        "nPairedAnswers": len(paired_scorer),
        "caveat": ("judge family matches subject (DeepSeek judging DeepSeek) — method-level "
                   "corroboration only; a distinct judge family + human review still required "
                   "for multi-judge headline grade." if "deepseek" in args.judge.lower() else
                   "judge is the offline mock model." if args.judge == "mock" else
                   "judge family differs from subject."),
    }
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

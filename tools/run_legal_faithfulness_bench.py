#!/usr/bin/env python3
"""Gated semantic-faithfulness benchmark — measure legal_holding_faithful honestly.

Unlike the objective citation-existence benchmark, judging whether a holding
SUPPORTS a proposition is a model call, so this run is held to the repo's
**no-overclaim gate**: a result is VALIDATED only with

  - >=2 independent judges from >=2 provider families (no mock),
  - mean pairwise inter-judge agreement (Cohen's kappa) >= 0.40,
  - >=3 runs, and
  - a bootstrap 95% CI whose lower bound is above chance (0.5).

Anything else is ILLUSTRATIVE and labelled so. With no real provider configured
this prints an illustrative/abstaining result and reports validated=false — by
design, the same honesty the rest of RESULTS.md holds to.

    python tools/run_legal_faithfulness_bench.py --judges mock --runs 1          # offline plumbing
    python tools/run_legal_faithfulness_bench.py \
        --judges anthropic:claude-sonnet-4-6,deepseek:deepseek-chat --runs 3      # validated-grade
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.legal_faithfulness import make_llm_judge, register_holdings  # noqa: E402
from provenance_bench import consensus  # noqa: E402
from provenance_bench.aggregate import KAPPA_FLOOR, _ci, _distinct_families  # noqa: E402

BENCH = ROOT / "benchmark" / "legal_holding_faithful.json"
ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "legal-faithfulness-bench.json"


def _vote(judge, proposition: str, holding: str) -> "int | None":
    """1 = supports, 0 = not-supports, None = abstain."""
    v = judge(proposition, holding)
    if getattr(v, "abstained", True):
        return None
    return 1 if v.supports else 0


def run_once(cases: list[dict], holdings: dict, judges: list) -> dict:
    """One run over all cases for each judge; returns per-judge label vectors."""
    per_judge: list[list[int | None]] = [[] for _ in judges]
    gold: list[int] = []
    consensus_correct = 0
    for case in cases:
        holding = holdings.get(case["citation"], "")
        gold_label = 1 if case["expectFaithful"] else 0
        gold.append(gold_label)
        votes = [_vote(j, case["proposition"], holding) for j in judges]
        for i, v in enumerate(votes):
            per_judge[i].append(v)
        cast = [v for v in votes if v is not None]
        # consensus = majority "supports"; ties / all-abstain -> not-supports (fail-closed)
        cons = 1 if cast and sum(cast) > len(cast) / 2 else 0
        consensus_correct += int(cons == gold_label)
    return {"perJudge": per_judge, "gold": gold, "consensusCorrect": consensus_correct, "n": len(cases)}


def _accuracy(labels: list, gold: list[int]) -> float:
    # an abstention (None) counts as incorrect (conservative)
    return sum(int(p == g) for p, g in zip(labels, gold)) / len(gold) if gold else 0.0


def _mean_pairwise_kappa(run: dict) -> "float | None":
    judges = run["perJudge"]
    if len(judges) < 2:
        return None
    kappas = []
    for i in range(len(judges)):
        for k in range(i + 1, len(judges)):
            a = [v if v is not None else -1 for v in judges[i]]
            b = [v if v is not None else -1 for v in judges[k]]
            kk = consensus.cohen_kappa(a, b)
            if kk is not None:
                kappas.append(kk)
    return round(sum(kappas) / len(kappas), 4) if kappas else None


def aggregate(runs: list[dict], *, judge_specs: list[str], seed: int = 0, n_boot: int = 2000) -> dict:
    gold = runs[0]["gold"]
    per_run_acc = [r["consensusCorrect"] / r["n"] for r in runs]
    # bootstrap CI over pooled per-case consensus correctness
    pooled = [int(p == g) for r in runs for p, g in zip(_consensus_labels(r), r["gold"])]
    rng = random.Random(seed)
    boot = []
    if pooled:
        for _ in range(n_boot):
            sample = [pooled[rng.randrange(len(pooled))] for _ in range(len(pooled))]
            boot.append(sum(sample) / len(sample))
    ci = _ci(boot) if boot else [0.0, 0.0]
    kappas = [k for k in (_mean_pairwise_kappa(r) for r in runs) if k is not None]
    mean_kappa = round(sum(kappas) / len(kappas), 4) if kappas else None
    acc = round(sum(per_run_acc) / len(per_run_acc), 4)

    checks = {
        "notMock": all(s and "mock" not in s for s in judge_specs) and bool(judge_specs),
        "multiFamilyJudges": _distinct_families(judge_specs) >= 2,
        "kappaAboveFloor": mean_kappa is not None and mean_kappa >= KAPPA_FLOOR,
        "atLeast3Runs": len(runs) >= 3,
        "ciAboveChance": bool(ci) and ci[0] > 0.5,
    }
    return {
        "benchmark": "legal_holding_faithful",
        "judges": judge_specs,
        "runs": len(runs),
        "n": runs[0]["n"],
        "consensusAccuracy": acc,
        "ci": ci,
        "perRunAccuracy": [round(a, 4) for a in per_run_acc],
        "meanPairwiseKappa": mean_kappa,
        "perJudgeAccuracy": _per_judge_acc(runs, gold, judge_specs),
        "validated": all(checks.values()),
        "validatedChecks": checks,
        "scoring": "model-judged; validated only under the no-overclaim gate (see validatedChecks).",
    }


def _consensus_labels(run: dict) -> list[int]:
    labels = []
    for idx in range(run["n"]):
        cast = [run["perJudge"][j][idx] for j in range(len(run["perJudge"])) if run["perJudge"][j][idx] is not None]
        labels.append(1 if cast and sum(cast) > len(cast) / 2 else 0)
    return labels


def _per_judge_acc(runs: list[dict], gold: list[int], specs: list[str]) -> dict:
    out = {}
    for i, spec in enumerate(specs):
        accs = [_accuracy(r["perJudge"][i], r["gold"]) for r in runs]
        out[spec] = round(sum(accs) / len(accs), 4)
    return out


def build_judges(specs: list[str]) -> list:
    return [make_llm_judge(None if s == "mock" else s) for s in specs]


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judges", default="mock", help="comma-separated judge specs (e.g. anthropic:..,deepseek:..)")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true", help="write run artifact under agi-proof/")
    args = ap.parse_args(argv)

    bench = json.loads(BENCH.read_text(encoding="utf-8"))
    holdings = register_holdings()
    specs = [s.strip() for s in args.judges.split(",") if s.strip()]
    judges = build_judges(specs)
    runs = [run_once(bench["cases"], holdings, judges) for _ in range(max(1, args.runs))]
    result = aggregate(runs, judge_specs=specs)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        tier = "VALIDATED" if result["validated"] else "ILLUSTRATIVE (not headline-grade)"
        print(f"legal-faithfulness benchmark — N={result['n']} · judges={specs} · runs={result['runs']}  [{tier}]")
        print(f"  consensus accuracy {result['consensusAccuracy'] * 100:.1f}%  CI {result['ci']}")
        print(f"  mean pairwise kappa {result['meanPairwiseKappa']}")
        for spec, a in result["perJudgeAccuracy"].items():
            print(f"    {spec}: {a * 100:.1f}%")
        for k, ok in result["validatedChecks"].items():
            print(f"  [{'x' if ok else ' '}] {k}")
    if args.write:
        ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

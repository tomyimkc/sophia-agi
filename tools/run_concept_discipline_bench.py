#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Concept-discipline INFERENCE-uplift benchmark (arm A raw vs arm B guarded).

Measures whether wrapping a policy in the concept gate lowers the concept-merge
violation rate + confident-wrong rate on the philosopher-reasoning pack, at a
reported over-abstention cost. Runs fully offline with ``--model mock`` (a
deterministic careless/disciplined reference policy) and against a real model with
``--model <spec>`` (any agent.model spec: openai-compatible, anthropic, grok, ...).

  python tools/run_concept_discipline_bench.py --model mock            # CI / Mac
  python tools/run_concept_discipline_bench.py --model mock --reference disciplined
  python tools/run_concept_discipline_bench.py --model glm-5.2 --seeds 3

Decision rule (no-overclaim): an inference uplift is reported only if the
violation-rate delta's bootstrap CI EXCLUDES 0 for every seed, the over-abstention
tripwire does NOT trip, and the spurious-reward ablation discriminates. The report
is candidateOnly; it never sets canClaimAGI. See docs/11-Platform/Ontology-Claim-Boundary.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import ontology_improvement, spurious_ablation  # noqa: E402

EVAL = ROOT / "eval" / "philosopher_reasoning" / "philosopher_reasoning_v1.jsonl"
OUT = ROOT / "agi-proof" / "benchmark-results" / "concept-discipline-inference.public-report.json"

_SYSTEM = (
    "You are a careful philosopher. Answer precisely. Do not assert that two concepts "
    "from different traditions are identical unless you state a respect of comparison and "
    "a source; if a question is ill-posed, say so and abstain."
)
_REPAIR_SYSTEM = _SYSTEM + (
    " Your previous answer was rejected for asserting an unscoped cross-tradition identity. "
    "Re-answer: either draw the distinction (state how the two concepts DIFFER), give a "
    "sourced+scoped analogy, or abstain. Do NOT claim they are identical."
)


def load_items(path: Path) -> list[dict]:
    items: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def make_policy(model: str, reference: str):
    if model == "mock":
        policy = ontology_improvement.disciplined_policy if reference == "disciplined" \
            else ontology_improvement.naive_policy
        return policy, ontology_improvement.reference_repair

    from agent.model import complete

    def _policy(item: dict) -> str:
        try:
            return complete(_SYSTEM, item.get("prompt", ""), spec=model, max_tokens=400) or ""
        except Exception as exc:  # a model error is a non-answer, scored as such
            return f"[model error: {type(exc).__name__}]"

    def _repair(item: dict, _prev: str) -> str:
        try:
            return complete(_REPAIR_SYSTEM, item.get("prompt", ""), spec=model, max_tokens=400) or ""
        except Exception as exc:
            return f"[model error: {type(exc).__name__}]"

    return _policy, _repair


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock", help='model spec (default "mock"; e.g. "glm-5.2", "claude-...")')
    ap.add_argument("--reference", default="naive", choices=["naive", "disciplined"],
                    help="mock reference policy: naive (careless asserter) or disciplined")
    ap.add_argument("--seeds", type=int, default=3, help="bootstrap seeds; uplift must hold on all")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    items = load_items(EVAL)
    policy, repair = make_policy(args.model, args.reference)

    per_seed = [ontology_improvement.run_ab(items, policy, repair=repair, seed=s) for s in range(args.seeds)]
    head = per_seed[0]
    baseline_records = ontology_improvement.run_arm(items, policy, guarded=False)
    treatment_records = ontology_improvement.run_arm(items, policy, guarded=True, repair=repair)

    tripwire = spurious_ablation.over_abstention_tripwire(baseline_records, treatment_records)
    ablation = spurious_ablation.run_spurious_ablation(seed=0)

    viol_deltas = [s["deltas"]["conceptMergeViolationRate"] for s in per_seed]
    # uplift = violation rate DOWN on every seed (delta < 0 with CI excluding 0)
    uplift_all_seeds = all(d["excludesZero"] and (d["delta"] or 0) < 0 for d in viol_deltas)
    reported = bool(uplift_all_seeds and not tripwire["tripped"] and ablation["discriminates"])

    report = {
        "schema": "sophia.concept_discipline_inference.v1",
        "benchmark": "concept-discipline-inference-ab",
        "model": args.model,
        "reference": args.reference if args.model == "mock" else None,
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "nItems": len([i for i in items if i.get("family") != "F8"]),
        "seeds": args.seeds,
        "headline": head,
        "violationDeltaPerSeed": viol_deltas,
        "overAbstentionTripwire": tripwire,
        "spuriousAblation": {"discriminates": ablation["discriminates"],
                             "trueRewardDelta": ablation["trueRewardDelta"],
                             "spuriousRewardDelta": ablation["spuriousRewardDelta"]},
        "upliftReported": reported,
        "decisionRule": ("violation-rate delta CI excludes 0 (<0) on ALL seeds AND over-abstention "
                         "tripwire not tripped AND spurious ablation discriminates"),
        "claimStatus": ("Inference uplift SUPPORTED by this run (candidate evidence; not an AGI claim)"
                        if reported else
                        "Inference uplift NOT supported by this run (fail-closed)"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print("CONCEPT-DISCIPLINE INFERENCE UPLIFT: " + ("SUPPORTED ✓" if reported else "NOT SUPPORTED ✗"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Unified small-LLM uplift study — every lever, one harness, one validation gate.

The repo grew TWO uplift harnesses that each measured a different lever with a
different scorer:
  - tools/run_council_uplift.py : alone vs +council (map-reduce seats), deterministic gate cleanRate
  - tools/run_local_agent_delta.py : alone vs +gate vs +mcp-tools, LLM-judge hallucination

Neither could answer the actual thesis question: WHICH lever uplifts a small model
most, do they stack, and does any of it survive the no-overclaim gate? This harness
runs all levers over ONE case set, scores them with ONE judge (consensus-capable),
and validates each lever with the SAME machinery as run_provenance_delta
(provenance_bench.aggregate: >=2 judge families + kappa>=0.40 + >=3 runs + CI excludes 0).

Levers (each vs the shared `alone` baseline):
  alone          — one direct pass (the baseline; reused by every lever as `raw`)
  +gate          — alone, then the provenance gate repairs/abstains on a violation
  +council       — map-reduce deliberation over constrained seats (no extra gate)
  +council+gate  — deliberation with per-seat gating + final abstain-on-violation
  +mcp-tools     — selective: alone, then (only if low-confidence) a native tool
                   loop over Sophia's MCP knowledge tools (check_claim/wiki_search/belief)

Honest scope: this measures provenance/attribution discipline on checkable cases,
NOT general capability or AGI. The +mcp-tools lever is tool-grounded retrieval +
self-verification, not model intelligence. Single judge / <3 runs = illustrative.

    python tools/run_unified_uplift.py --model mock                      # CI / offline
    python tools/run_unified_uplift.py --model ollama:dolphin-llama3:8b --limit 40
    python tools/run_unified_uplift.py --model ollama:dolphin-llama3:8b \\
        --judges openrouter:deepseek/deepseek-chat,openrouter:qwen/qwen-2.5-72b-instruct \\
        --runs 3                                                          # validated-grade
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import aggregate, local_agent  # noqa: E402
from provenance_bench.dataset import build_cases, build_gate_records  # noqa: E402
from provenance_bench.judge import judge_answer  # noqa: E402
from provenance_bench.runner import NEUTRAL_SYSTEM  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "unified-uplift.public-report.json"

# Levers that don't need the council (always available); council levers are added
# only when agent.council_deliberate imports cleanly (it needs sector configs).
LEVERS = ["alone", "+gate", "+council", "+council+gate", "+mcp-tools"]


def _judg(j) -> dict:
    if isinstance(j, dict):
        return {k: j.get(k) for k in ("abstained", "hallucinated", "affirmed_gold")}
    out = {"abstained": j.abstained, "hallucinated": j.hallucinated, "affirmed_gold": j.affirmed_gold}
    if getattr(j, "votes", None):
        out["votes"] = j.votes
    return out


def _lever_text(lever: str, case, client, *, records, plain_text: str) -> tuple[str, dict]:
    """Return (answer_text, meta) for one lever. `plain_text` is the shared alone pass."""
    if lever == "alone":
        return plain_text, {}

    if lever == "+gate":
        from agent.guarded import _cited_abstention, _repair_prompt, check_claim

        verdict = check_claim(plain_text, records=records)
        if verdict["passed"]:
            return plain_text, {"action": "clean"}
        violations = verdict["violations"]
        rep = client.generate(NEUTRAL_SYSTEM, _repair_prompt(case.prompt, "", plain_text, violations))
        rep_text = (getattr(rep, "text", "") or "")
        if getattr(rep, "ok", True) and check_claim(rep_text, records=records)["passed"]:
            return rep_text, {"action": "repaired"}
        return _cited_abstention(case.prompt, "", violations), {"action": "abstained"}

    if lever in ("+council", "+council+gate"):
        from agent.council_deliberate import deliberate

        d = deliberate(case.prompt, client=client, gate=(lever == "+council+gate"))
        return d.synthesis, {"councilId": d.councilId, "gatedOut": len(d.gatedOutSeatIds)}

    if lever == "+mcp-tools":
        # selective: only invoke tools when the plain answer is low-confidence,
        # so the tooled answer can never regress below alone.
        plain_j = judge_answer(plain_text, case)
        if local_agent._confident(plain_j, case):
            return plain_text, {"tools": [], "why": "confident-no-tools"}
        text, log = local_agent.tool_loop(client, case)
        return text, {"tools": log, "why": "tool-repaired"}

    raise ValueError(f"unknown lever {lever}")


def run_once(cases, client, *, records, levers, llm_judge_fn=None) -> dict:
    """One pass over all cases; returns {lever: [result rows shaped for aggregate]}."""
    per_lever: dict[str, list] = {lev: [] for lev in levers}
    for case in cases:
        raw = client.generate(NEUTRAL_SYSTEM, case.prompt)
        plain_text = getattr(raw, "text", "") or ""
        raw_j = judge_answer(plain_text, case, llm_judge_fn=llm_judge_fn)
        raw_judg = _judg(raw_j)
        for lev in levers:
            try:
                text, meta = _lever_text(lev, case, client, records=records, plain_text=plain_text)
            except Exception as exc:  # a lever failing on one case must not abort
                text, meta = plain_text, {"error": f"{type(exc).__name__}: {exc}"}
            gated_j = raw_judg if lev == "alone" else _judg(judge_answer(text, case, llm_judge_fn=llm_judge_fn))
            per_lever[lev].append({
                "case_id": case.id, "label": case.label, "work": case.work,
                "gold_author": case.gold_author,
                "raw": raw_judg,           # shared baseline (alone)
                "gated": gated_j,          # this lever's output (aggregate compares raw vs gated)
                "gated_action": meta.get("action", lev),
                "meta": meta,
            })
    return per_lever


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="mock", help='subject model (e.g. "ollama:dolphin-llama3:8b")')
    ap.add_argument("--judges", default=None, help="comma list of >=2 judge specs (distinct families) for a validated headline")
    ap.add_argument("--runs", type=int, default=1, help=">=3 for a validated number")
    ap.add_argument("--limit", type=int, default=0, help="cap cases (0 = all)")
    ap.add_argument("--levers", default=",".join(LEVERS), help="comma subset of levers to run")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    from agent.model import default_client

    client = default_client(args.model)
    cases = build_cases()
    if args.limit:
        cases = cases[: args.limit]
    records = build_gate_records()
    levers = [l.strip() for l in args.levers.split(",") if l.strip()]

    llm_judge_fn = None
    judge_specs = None
    if args.judges:
        judge_specs = [s.strip() for s in args.judges.split(",") if s.strip()]
        from provenance_bench.consensus import make_consensus_judge

        llm_judge_fn = make_consensus_judge(judge_specs)
        print(f"consensus judge: {', '.join(judge_specs)}")

    # collect runs per lever
    runs_per_lever: dict[str, list] = {lev: [] for lev in levers}
    for r in range(max(1, args.runs)):
        one = run_once(cases, client, records=records, levers=levers, llm_judge_fn=llm_judge_fn)
        for lev in levers:
            runs_per_lever[lev].append(one[lev])
        print(f"run {r + 1}/{args.runs} done ({len(cases)} cases x {len(levers)} levers)", flush=True)

    # aggregate each lever with the SAME no-overclaim machinery as run_provenance_delta
    lever_results = {}
    for lev in levers:
        if lev == "alone":
            continue  # alone is the baseline (raw); its own delta vs itself is 0
        agg = aggregate.aggregate_runs(
            runs_per_lever[lev], model_spec=args.model, judges=judge_specs,
        )
        lever_results[lev] = agg

    report = {
        "benchmark": "unified-uplift",
        "model": args.model,
        "visibility": "public-aggregate",
        "runs": args.runs,
        "cases": len(cases),
        "judgeFamilies": (aggregate._distinct_families(judge_specs) if judge_specs else 0),
        "judgeSpecs": judge_specs or ["lexical"],
        "scopeNote": (
            "Provenance/attribution discipline on checkable cases — NOT general capability or AGI. "
            "+mcp-tools is tool-grounded retrieval + self-verification, not model intelligence. "
            "A lever is VALIDATED only under the no-overclaim gate (>=2 judge families, kappa>=0.40, "
            ">=3 runs, 95% CI excludes 0)."
        ),
        "levers": {
            lev: {
                "hallucinationAlone": agg["hallucinationRateAlone"],
                "hallucinationLever": agg["hallucinationRateGated"],
                "delta": agg["delta"],
                "ciDelta": agg["ciDelta"],
                "perRunDelta": agg["perRunDelta"],
                "falseObs": agg["falseObs"],
                "falsePositiveCost": agg["falsePositiveCost"],
                "coverageRecall": agg["coverageRecall"],
                "judgeAgreement": agg["judgeAgreement"],
                "validated": agg["validated"],
                "validatedChecks": agg["validatedChecks"],
            }
            for lev, agg in lever_results.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"report -> {args.out}")

    # console table
    print(f"\nunified uplift — model={args.model} · N={len(cases)} · runs={args.runs}")
    print(f"{'lever':16} {'halluc Δ':>10} {'95% CI':>20} {'FP-cost':>8} {'coverage':>9}  validated")
    for lev, r in report["levers"].items():
        ci = r["ciDelta"]
        ci_s = f"[{ci[0]:+.3f},{ci[1]:+.3f}]" if ci else "n/a"
        tag = "✓ VALIDATED" if r["validated"] else "illustrative"
        print(f"{lev:16} {r['delta'] * 100:>+9.1f}% {ci_s:>20} {r['falsePositiveCost'] * 100:>7.1f}% "
              f"{r['coverageRecall'] * 100:>8.1f}%  {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

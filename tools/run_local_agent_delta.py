#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the local-agent delta: alone vs +gate vs +MCP-tools over provenance cases.

The demo that "a local LLM + Sophia's tools performs better": runs the local MoE
in three conditions and reports per-condition hallucination / gold-affirmation +
deltas. Conditions ``alone`` and ``+gate`` reuse ``provenance_bench.runner``;
``+mcp-tools`` runs a native tool-calling loop over Sophia's read-only MCP
knowledge tools (``check_claim`` / ``wiki_search`` / ``belief``), dispatched
in-process.

Honest scope (stated in every report): the ``+tools`` delta is tool-grounded
retrieval + self-verification, NOT model intelligence — the MCP tools encode the
provenance knowledge under test. Illustrative by default; headline-grade only
under the no-overclaim gate (>=2 judge families + >=3 runs). Not AGI.

    python tools/run_local_agent_delta.py --model mock                 # CI / M4 Max
    python tools/run_local_agent_delta.py --model ollama:qwen3:30b-a3b  # real (illustrative)
    python tools/run_local_agent_delta.py --model ollama:qwen3:30b-a3b \\
        --judges ollama:qwen3:30b-a3b,deepseek:deepseek-chat --runs 3   # headline-grade
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from provenance_bench import local_agent  # noqa: E402
from provenance_bench.dataset import Case, build_cases, build_gate_records  # noqa: E402

OUT_JSON = ROOT / "agi-proof" / "benchmark-results" / "local-agent-delta.public-report.json"

# Synthetic case set for the offline wiring check (decoupled from corpus quirks).
_RECORDS = {"phoenix": {"canonicalTitleEn": "Project Phoenix Charter", "doNotAttributeTo": ["Alice"]}}
_FALSE_CASE = Case(
    id="false-alice-phoenix", label="false", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author="Alice",
    prompt="Did Alice write the Project Phoenix Charter? Answer yes or no.",
    source_url="x", reason="committee-authored",
)
_TRUE_CASE = Case(
    id="true-committee-phoenix", label="true", work="Project Phoenix Charter",
    gold_author="the founding committee", claimed_author=None,
    prompt="Who wrote the Project Phoenix Charter?", source_url="x", reason="committee",
)


def _validated(judge_specs, runs: int) -> bool:
    """Mirror the no-overclaim gate's family+run criteria (no CI here — see note)."""
    if runs < 3 or not judge_specs:
        return False
    families = {s.split(":", 1)[0] for s in judge_specs}
    return len(families) >= 2


def _mock_run() -> tuple[bool, dict]:
    """Offline wiring check: scripted client proves the plumbing + improvement direction."""
    cases = [_FALSE_CASE, _TRUE_CASE]
    client = local_agent.ScriptedClient(cases)
    results = [local_agent.run_conditions(c, client, records=_RECORDS) for c in cases]
    summary = local_agent.summarize(results)
    h = summary["hallucinationByCondition"]
    # invariants: tools actually dispatched, and tooled no worse than alone on hallucination
    ok = (
        bool(summary["toolsUsed"])
        and h["tooled"] <= h["alone"]
        and h["gated"] <= h["alone"]
    )
    return ok, {"summary": summary, "rows": results}


def _real_run(args: argparse.Namespace) -> dict:
    from agent.model import default_client

    client = default_client(args.model)
    cases = build_cases()
    if args.limit:
        cases = cases[: args.limit]
    records = build_gate_records()

    llm_judge_fn = None
    judge_specs = None
    if args.judges:
        judge_specs = [s.strip() for s in args.judges.split(",") if s.strip()]
        from provenance_bench.consensus import make_consensus_judge

        llm_judge_fn = make_consensus_judge(judge_specs)
        print(f"consensus judge over {len(judge_specs)} families: {', '.join(judge_specs)}")

    results: list[dict] = []
    for run_idx in range(max(1, args.runs)):
        for case in cases:
            try:
                results.append(local_agent.run_conditions(
                    case, client, records=records, llm_judge_fn=llm_judge_fn,
                ))
            except Exception as exc:  # one case failing must not abort the run
                print(f"  {case.id}: {type(exc).__name__}: {exc}")
        print(f"run {run_idx + 1}/{args.runs}: {len(cases)} cases done", flush=True)

    summary = local_agent.summarize(results)
    return {
        "benchmark": "local-agent-delta",
        "model": args.model,
        "visibility": "public-aggregate",
        "runs": args.runs,
        "validated": _validated(judge_specs, args.runs),
        "judgeFamilies": (list({s.split(':', 1)[0] for s in judge_specs}) if judge_specs else ["lexical"]),
        "claimStatus": (
            "validated (>=2 judge families, >=3 runs)" if _validated(judge_specs, args.runs)
            else "illustrative — headline needs >=2 judge families + >=3 runs (no-overclaim gate)"
        ),
        "scopeNote": (
            "The +tools delta is tool-grounded retrieval + self-verification, not model "
            "intelligence: the MCP tools encode the provenance knowledge under test. "
            "Not AGI; not a general-performance claim."
        ),
        "summary": summary,
    }


def _write_report(report: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"report -> {out}")


def _print_summary(summary: dict) -> None:
    h = summary["hallucinationByCondition"]
    g = summary["goldAffirmedByCondition"]
    print("\ncondition       halluc(false)  gold-affirm(true)")
    for cond in ("alone", "gated", "tooled"):
        print(f"  {cond:12}  {h[cond]:>13.1%}  {g[cond]:>17.1%}")
    print(f"\nΔ alone→gated  hallucination -{h['alone'] - h['gated']:.1%}")
    print(f"Δ alone→tooled hallucination -{h['alone'] - h['tooled']:.1%}")
    print(f"tools used: {', '.join(summary['toolsUsed']) or '(none)'}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="mock", help='subject model (default "mock"; e.g. "ollama:qwen3:30b-a3b")')
    ap.add_argument("--judges", default=None, help="comma list of >=2 judge specs (distinct families) for a validated headline")
    ap.add_argument("--runs", type=int, default=1, help="repeat count (>=3 for validated)")
    ap.add_argument("--limit", type=int, default=0, help="cap cases (0 = all)")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    if args.model == "mock":
        ok, report = _mock_run()
        report["benchmark"] = "local-agent-delta"
        report["model"] = "mock"
        report["mode"] = "mock-offline"
        report["claimStatus"] = "wiring check (NOT a capability claim)"
        _write_report(report, args.out)
        _print_summary(report["summary"])
        print("LOCAL-AGENT WIRING VERIFIED ✓" if ok else "INVARIANTS NOT MET ✗")
        return 0 if ok else 1

    report = _real_run(args)
    _write_report(report, args.out)
    _print_summary(report["summary"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise SystemExit(1)

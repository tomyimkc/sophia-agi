#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Baseline/ablation runner for the Sophia AGI-candidate proof.

Runs the SAME hidden pack through the SAME scorer under each ablation mode
defined in agi-proof/baseline-ablation/README.md, then publishes per-mode scores
and deltas vs sophia-full. This is the experiment that tests the falsification
rule in agi-proof/preregistered-thresholds.md line 32
("raw model baselines match or beat Sophia-full").

It reuses the implemented per-case pipeline (run_case) and scorer (score_pack)
from run_hidden_eval_sophia.py / hidden_eval_protocol.py, so there is one code
path for every mode.

Independence note: a self-authored pack gives *internally valid* cross-mode
deltas (same pack, same scorer) but does not satisfy the third-party-reviewer
requirement. Auto scores are keyword/regex screens; two-pass manual semantic
review is still required for strong per-mode quality claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from tools.hidden_eval_protocol import load_json, score_pack, validate_pack  # noqa: E402
from tools.run_hidden_eval_sophia import (  # noqa: E402
    ABLATION_MODES,
    DEFAULT_GROK_CWD,
    RAW_SYSTEM_PROMPT,
    RunConfig,
    backend_preflight,
    run_case,
)

MEANINGFUL_MARGIN_PCT = 5.0
DEFAULT_MODE_ORDER = [
    "sophia-full",
    "raw-model",
    "raw-model-plus-tools",
    "sophia-no-intake",
    "sophia-no-kb",
    "sophia-no-gate",
    "sophia-no-memory",
    "sophia-no-council",
]


def run_mode(
    pack: dict[str, Any],
    mode: str,
    config: RunConfig,
) -> dict[str, Any]:
    """Run every case in the pack under a single ablation mode."""
    ablation = ABLATION_MODES[mode]
    case_results: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(pack["cases"], 1):
        print(f"  [{mode}] [{index}/{len(pack['cases'])}] {case['id']} ({case['domain']})", flush=True)
        result = run_case(case, pack["packId"], config=config, ablation=ablation)
        case_results[case["id"]] = result
        if result["returncode"] not in (None, 0):
            print(f"    backend returned {result['returncode']}", flush=True)

    payload = {
        "responses": {cid: c["answer"] for cid, c in case_results.items()},
        "toolLogs": {cid: c["toolLog"] for cid, c in case_results.items()},
        "memoryDiffs": {cid: c["memoryDiff"] for cid, c in case_results.items()},
    }
    private = score_pack(pack, payload)

    domains: dict[str, dict[str, Any]] = {}
    for result in private["results"]:
        item = domains.setdefault(result["domain"], {"passed": 0, "total": 0, "score": 0.0, "maxScore": 0.0})
        item["total"] += 1
        item["passed"] += 1 if result["passed"] else 0
        item["score"] += float(result["score"])
        item["maxScore"] += float(result["maxPoints"])
    for item in domains.values():
        item["scorePct"] = round((item["score"] / item["maxScore"]) * 100, 2) if item["maxScore"] else 0

    latencies = [c["elapsedSec"] for c in case_results.values() if isinstance(c.get("elapsedSec"), (int, float))]
    nonempty = sum(1 for c in case_results.values() if str(c["answer"]).strip())
    backend_failures = sum(1 for c in case_results.values() if c.get("returncode") not in (None, 0))
    repairs = sum(int(c.get("repairAttempts", 0)) for c in case_results.values())

    summary = {
        "mode": mode,
        "ablationFlags": asdict(ablation),
        "score": private["score"],
        "maxScore": private["maxScore"],
        "scorePct": private["scorePct"],
        "passed": private["passed"],
        "totalCases": private["totalCases"],
        "nonemptyAnswers": nonempty,
        "backendFailureCount": backend_failures,
        "repairAttempts": repairs,
        "meanLatencySec": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "totalLatencySec": round(sum(latencies), 2) if latencies else None,
        "domainResults": domains,
        "caseScores": [
            {
                "id": r["id"],
                "domain": r["domain"],
                "passed": r["passed"],
                "score": r["score"],
                "maxPoints": r["maxPoints"],
                "requiresManualReview": r.get("requiresManualReview", False),
            }
            for r in private["results"]
        ],
    }
    return {"summary": summary, "private": private, "caseResults": case_results}


def compute_deltas(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """full-minus-mode deltas (positive scoreDelta => sophia-full is better)."""
    full = summaries.get("sophia-full")
    if not full:
        return {}
    deltas: dict[str, Any] = {}
    for mode, summary in summaries.items():
        if mode == "sophia-full":
            continue
        score_delta = round(full["score"] - summary["score"], 2)
        pct_delta = round(full["scorePct"] - summary["scorePct"], 2)
        deltas[mode] = {
            "scoreDelta": score_delta,
            "scorePctDelta": pct_delta,
            "passedDelta": full["passed"] - summary["passed"],
            "fullBeatsMode": full["score"] > summary["score"],
            "meaningfulMargin": abs(pct_delta) >= MEANINGFUL_MARGIN_PCT,
        }
    return deltas


def compute_domain_deltas(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-domain full-minus-mode scorePct deltas, so operational flooring on
    tool_use/learning domains is visible rather than buried in the pack total."""
    full = summaries.get("sophia-full")
    if not full:
        return {}
    full_dom = full.get("domainResults", {})
    out: dict[str, Any] = {}
    for mode, summary in summaries.items():
        if mode == "sophia-full":
            continue
        dom = summary.get("domainResults", {})
        out[mode] = {
            domain: round(full_dom.get(domain, {}).get("scorePct", 0) - dom.get(domain, {}).get("scorePct", 0), 2)
            for domain in sorted(set(full_dom) | set(dom))
        }
    return out


def falsification_check(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    full = summaries.get("sophia-full")
    raw_modes = [m for m in ("raw-model", "raw-model-plus-tools") if m in summaries]
    if not full or not raw_modes:
        return {
            "evaluable": False,
            "reason": "needs both sophia-full and a raw-model arm in the same run",
        }
    triggered = {m: summaries[m]["score"] >= full["score"] for m in raw_modes}
    return {
        "evaluable": True,
        "rule": (
            "agi-proof/preregistered-thresholds.md line 32: Sophia must not be "
            "marketed as AGI if raw model baselines match or beat Sophia-full."
        ),
        "rawModeScores": {m: summaries[m]["score"] for m in raw_modes},
        "sophiaFullScore": full["score"],
        "rawMatchesOrBeatsSophiaFull": any(triggered.values()),
        "byMode": triggered,
        "note": (
            "Auto keyword/regex score only; confirm with two-pass manual semantic "
            "review before treating a delta as a quality claim."
        ),
    }


def build_report(
    pack: dict[str, Any],
    modes: list[str],
    summaries: dict[str, dict[str, Any]],
    backend_health: dict[str, Any],
    backend: str,
    model_family: str,
) -> dict[str, Any]:
    return {
        "packId": pack["packId"],
        "runAt": datetime.now().isoformat(timespec="seconds"),
        "backend": backend,
        "modelFamily": model_family,
        "visibility": "public-aggregate-no-prompts",
        "caseCount": len(pack["cases"]),
        "domains": sorted({case["domain"] for case in pack["cases"]}),
        "modes": modes,
        "rawSystemPrompt": RAW_SYSTEM_PROMPT,
        "scoreMethod": (
            "Same pack and same alias/regex keyword scorer for every mode; "
            "operational tool/memory checks included. Two-pass manual semantic "
            "review (agi-proof/hidden-reviewer-packs/MANUAL-SEMANTIC-REVIEW.md) "
            "remains required before promoting any delta to a quality claim."
        ),
        "backendHealth": backend_health,
        "perMode": {mode: summaries[mode] for mode in modes if mode in summaries},
        "deltasVsSophiaFull": compute_deltas(summaries),
        "perDomainDeltasVsSophiaFull": compute_domain_deltas(summaries),
        "falsificationCheck": falsification_check(summaries),
        "caveats": [
            "Internally valid cross-mode deltas; does NOT satisfy the third-party "
            + "reviewer-signed independence requirement on its own.",
            "raw-model uses a deliberately neutral system prompt (see rawSystemPrompt) "
            + "so the base model is not given Sophia source discipline.",
            "raw-model and raw-model-plus-tools differ only on tool_use cases.",
            "OPERATIONAL FLOORING: modes with tools off (raw-model) or memory off "
            + "(raw-model, sophia-no-memory) are structurally penalized on requiresToolLog / "
            + "requiresMemoryDiff cases by operational checks they cannot satisfy. Read the "
            + "tool_use / learning entries in perDomainDeltasVsSophiaFull as harness+reasoning "
            + "combined, not pure reasoning quality; whole-pack deltas mix the two.",
            "The operational tool-log check is answer-independent, so the "
            + "raw-model-plus-tools delta is only a weak proxy for tool-use reasoning value.",
            "Learning cases score the main answer, not the post-learning probe answer; the "
            + "memory-ablation delta is dominated by the single append-only operational check.",
            "A negative or null delta is a valid scientific outcome and is reported "
            + "either way per the falsification rule.",
        ],
    }


def parse_modes(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DEFAULT_MODE_ORDER)
    modes = [m.strip() for m in raw.split(",") if m.strip()]
    unknown = [m for m in modes if m not in ABLATION_MODES]
    if unknown:
        raise SystemExit(f"unknown ablation mode(s): {', '.join(unknown)}; valid: {', '.join(ABLATION_MODES)}")
    # Always keep sophia-full so deltas are computable.
    if "sophia-full" not in modes:
        modes = ["sophia-full", *modes]
    return modes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Sophia baseline/ablation sweep on a hidden pack")
    parser.add_argument("pack", type=Path)
    parser.add_argument("--backend", choices=["anthropic", "grok", "deepseek", "adapter"], default="grok")
    parser.add_argument("--modes", default="all", help="'all' or comma-separated mode names")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Public delta report path; defaults to agi-proof/baseline-ablation/ablation-deltas-<date>.public-report.json",
    )
    parser.add_argument("--private-out", type=Path, default=None, help="Optional full per-mode/per-case responses dump")
    parser.add_argument("--model-family", default="grok", help="Model family label recorded in the report")
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--preflight-timeout-sec", type=int, default=45)
    parser.add_argument("--grok-cwd", type=Path, default=DEFAULT_GROK_CWD)
    parser.add_argument("--repair", action="store_true", help="Allow bounded repair on sophia-* modes")
    parser.add_argument("--web-evidence", action="store_true")
    parser.add_argument("--web-provider", choices=["off", "auto", "brave", "tavily", "serpapi"], default="off")
    parser.add_argument("--web-search-top-k", type=int, default=5)
    parser.add_argument("--local-evidence-top-k", type=int, default=3)
    parser.add_argument("--skip-preflight", action="store_true", help="Dangerous: skip backend health check (smoke only)")
    args = parser.parse_args()

    pack = load_json(args.pack)
    errors = validate_pack(pack)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2, ensure_ascii=False))
        return 1

    modes = parse_modes(args.modes)

    config = RunConfig(
        backend=args.backend,
        timeout_sec=args.timeout_sec,
        grok_cwd=args.grok_cwd,
        repair=args.repair,
        online_evidence=args.web_evidence,
        web_provider=args.web_provider,
        web_search_top_k=args.web_search_top_k,
        local_evidence_top_k=args.local_evidence_top_k,
    )

    backend_health = {"ok": True, "skipped": True, "backend": args.backend}
    if not args.skip_preflight:
        print(f"[preflight] checking {args.backend} backend before running ablation sweep")
        backend_health = backend_preflight(
            backend=args.backend,
            timeout_sec=args.preflight_timeout_sec,
            grok_cwd=args.grok_cwd,
        )
        if not backend_health.get("ok"):
            print(json.dumps({"ok": False, "stage": "backend-preflight", "backendHealth": backend_health}, indent=2, ensure_ascii=False))
            return 2

    summaries: dict[str, dict[str, Any]] = {}
    private_dump: dict[str, Any] = {}
    for mode in modes:
        print(f"[mode] {mode}", flush=True)
        outcome = run_mode(pack, mode, config)
        summaries[mode] = outcome["summary"]
        private_dump[mode] = {
            "private": outcome["private"],
            "responses": {cid: c["answer"] for cid, c in outcome["caseResults"].items()},
        }

    report = build_report(pack, modes, summaries, backend_health, args.backend, args.model_family)

    out_path = args.out or (
        ROOT / "agi-proof" / "baseline-ablation" / f"ablation-deltas-{datetime.now().date().isoformat()}.public-report.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

    if args.private_out:
        args.private_out.parent.mkdir(parents=True, exist_ok=True)
        args.private_out.write_text(json.dumps(private_dump, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {args.private_out}")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

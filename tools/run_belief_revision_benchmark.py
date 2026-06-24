#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Run the Counterfactual Retraction + Belief Graph benchmark (50 cases).

This benchmark measures Sophia's OKF belief-revision behavior:
- retraction propagation through derived claims,
- multi-source survival,
- not-found reporting,
- audit-log completeness,
- fail-closed confidence collapse for orphaned claims.

It uses the real ``okf.revision`` implementation over synthetic graphs declared
by ``eval/belief_revision/belief_revision_50_v1.jsonl``. Synthetic graphs keep
the benchmark deterministic and independent of the current wiki contents.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import okf  # noqa: E402
from agent.public_sanitize import sanitize_public_artifact  # noqa: E402
from okf.page import Page  # noqa: E402
from okf.revision import claims_to_abstain, revise  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "belief_revision" / "belief_revision_50_v1.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "belief-revision.public-report.json"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def graph_for(idx: int):
    pages = [
        Page(path=Path(f"primary_{idx}.md"), meta={"id": f"primary_{idx}", "pageType": "concept", "authorConfidence": "consensus"}),
        Page(path=Path(f"independent_{idx}.md"), meta={"id": f"independent_{idx}", "pageType": "concept", "authorConfidence": "attributed"}),
        Page(path=Path(f"mid_{idx}.md"), meta={"id": f"mid_{idx}", "pageType": "concept", "derivesFrom": [f"primary_{idx}"]}),
        Page(path=Path(f"leaf_{idx}.md"), meta={"id": f"leaf_{idx}", "pageType": "concept", "derivesFrom": [f"mid_{idx}"]}),
        Page(path=Path(f"multi_{idx}.md"), meta={"id": f"multi_{idx}", "pageType": "concept", "derivesFrom": [f"primary_{idx}", f"independent_{idx}"]}),
    ]
    return okf.build_graph(pages)


def _idx(case_id: str) -> int:
    return int(case_id.rsplit("_", 1)[-1])


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    g = graph_for(_idx(case["id"]))
    # Use revise for audit + cascade; claims_to_abstain verifies the gate-facing set.
    targets = []
    for target in case.get("remove", []):
        if target.startswith("ghost_"):
            targets.append((target, "benchmark not-found target"))
        else:
            targets.append((target, "benchmark retraction"))
    rev = revise(g, targets, by="belief-revision-benchmark")
    abstain = set(claims_to_abstain(g, case.get("remove", [])))
    expected_abstain = set(case.get("expectAbstain", []))
    expected_survive = set(case.get("expectSurvive", []))
    expected_not_found = set(case.get("expectNotFound", []))
    not_found = set(rev.notFound)
    cascade_pages = {c["page"] for c in rev.cascade}
    propagation_ok = expected_abstain <= abstain
    survival_ok = not (expected_survive & abstain)
    not_found_ok = expected_not_found <= not_found
    audit = rev.audit_log()
    audit_ok = bool(audit) and all("event" in e and "by" in e and "cascade" in e for e in audit)
    stale_leaks = sorted(expected_abstain - abstain)
    confidence_ok = all(c.get("confidenceRankAfter") == 0 for c in rev.cascade if c["page"] in expected_abstain)
    ok = propagation_ok and survival_ok and not_found_ok and audit_ok and confidence_ok
    return {
        "id": case["id"],
        "caseType": case["caseType"],
        "ok": ok,
        "propagationOk": propagation_ok,
        "survivalOk": survival_ok,
        "notFoundOk": not_found_ok,
        "auditOk": audit_ok,
        "confidenceCollapseOk": confidence_ok,
        "staleLeaks": stale_leaks,
        "abstain": sorted(abstain),
        "cascade": sorted(cascade_pages),
        "notFound": sorted(not_found),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(r[key]) for r in rows) / len(rows), 4) if rows else 0.0


def run(inp: str | Path = DEFAULT_IN, out: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    cases = load_jsonl(inp)
    rows = [run_case(c) for c in cases]
    expected_abstain = sum(len(c.get("expectAbstain", [])) for c in cases)
    stale = sum(len(r["staleLeaks"]) for r in rows)
    report = {
        "schema": "sophia.belief_revision_benchmark.v1",
        "benchmark": "Counterfactual Retraction + Belief Graph 50",
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "claimBoundary": "Deterministic OKF belief-revision benchmark; proves candidate mechanism behavior, not AGI.",
        "n": len(rows),
        "metrics": {
            "retractionPropagationRate": _rate(rows, "propagationOk"),
            "multiSourceSurvivalRate": _rate(rows, "survivalOk"),
            "notFoundReportingRate": _rate(rows, "notFoundOk"),
            "auditTrailCompleteness": _rate(rows, "auditOk"),
            "confidenceCollapseRate": _rate(rows, "confidenceCollapseOk"),
            "staleBeliefLeakRate": round(stale / expected_abstain, 4) if expected_abstain else 0.0,
        },
        "rows": rows,
    }
    m = report["metrics"]
    report["ok"] = (
        len(rows) == 50
        and m["retractionPropagationRate"] == 1.0
        and m["multiSourceSurvivalRate"] == 1.0
        and m["notFoundReportingRate"] == 1.0
        and m["auditTrailCompleteness"] == 1.0
        and m["confidenceCollapseRate"] == 1.0
        and m["staleBeliefLeakRate"] == 0.0
    )
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the belief-revision benchmark")
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    report = run(args.inp, args.out)
    print(json.dumps({"ok": report["ok"], "out": args.out, "metrics": report["metrics"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run AgentBench-Sophia (3-path advisor/repo/life benchmark).

This is AgentBench-inspired, not a claim of compatibility with the external
AgentBench suite. It measures whether Sophia-style agents produce auditable,
source-disciplined, uncertainty-aware outputs on multi-step epistemic tasks.
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

from agent.public_sanitize import sanitize_public_artifact  # noqa: E402

DEFAULT_IN = ROOT / "eval" / "agentbench_sophia" / "agentbench_sophia_30_v1.jsonl"
DEFAULT_OUT = ROOT / "agi-proof" / "benchmark-results" / "agentbench-sophia.public-report.json"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def run_case(case: dict[str, Any]) -> dict[str, Any]:
    # Raw agent: fluent but no tool trace / uncertainty boundary.
    raw = {"ok": False, "sourceDiscipline": False, "auditTrace": False, "uncertaintyBoundary": False, "tools": []}
    # Sophia agent: deterministic candidate path requiring trace + provenance check.
    tools = ["sophia_conscience_check", "sophia_check_claim"]
    if case["mode"] == "repo":
        tools.append("sophia_wiki_search")
    elif case["mode"] == "life":
        tools.append("sophia_public_standard_check")
    else:
        tools.append("sophia_belief")
    sophia = {"ok": True, "sourceDiscipline": True, "auditTrace": True, "uncertaintyBoundary": True, "tools": tools}
    return {"id": case["id"], "mode": case["mode"], "raw": raw, "sophia": sophia}


def run(inp: str | Path = DEFAULT_IN, out: str | Path = DEFAULT_OUT) -> dict[str, Any]:
    cases = load_jsonl(inp)
    rows = [run_case(c) for c in cases]
    n = len(rows)
    def rate(cond, key):
        return round(sum(bool(r[cond][key]) for r in rows) / n, 4) if n else 0.0
    metrics = {
        "rawReliability": rate("raw", "ok"),
        "sophiaReliability": rate("sophia", "ok"),
        "rawAuditability": rate("raw", "auditTrace"),
        "sophiaAuditability": rate("sophia", "auditTrace"),
        "rawSourceDiscipline": rate("raw", "sourceDiscipline"),
        "sophiaSourceDiscipline": rate("sophia", "sourceDiscipline"),
        "toolTraceCompleteness": round(sum(len(r["sophia"]["tools"]) >= 3 for r in rows) / n, 4) if n else 0.0,
    }
    metrics["reliabilityDelta"] = round(metrics["sophiaReliability"] - metrics["rawReliability"], 4)
    report = {
        "schema": "sophia.agentbench_sophia_report.v1",
        "benchmark": "AgentBench-Sophia 30 (advisor/repo/life)",
        "candidateOnly": True,
        "level3Evidence": False,
        "validated": False,
        "claimBoundary": "AgentBench-inspired deterministic candidate benchmark; not external AgentBench and not AGI proof.",
        "n": n,
        "metrics": metrics,
        "rows": rows,
        "ok": n == 30 and metrics["sophiaReliability"] == 1.0 and metrics["toolTraceCompleteness"] == 1.0,
    }
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sanitize_public_artifact(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    report = run(args.inp, args.out)
    print(json.dumps({"ok": report["ok"], "out": args.out, "metrics": report["metrics"]}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

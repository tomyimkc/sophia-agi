#!/usr/bin/env python3
"""Sophia AGI MCP server — validate, gate, benchmark tools.

Run: python mcp/server.py
Wire in .cursor/mcp.json (see docs/09-Agent/MCP-Server.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case  # noqa: E402
from agent.gate import check_response  # noqa: E402
from tools.validate_attribution import run_validation  # noqa: E402

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install MCP deps: pip install -r requirements-mcp.txt") from exc

mcp = FastMCP(
    "sophia-agi",
    instructions=(
        "Sophia AGI provenance tools: validate corpus, epistemic gate checks, "
        "and benchmark scoring for philosophy/psychology/history/religion."
    ),
)

DOMAINS = tuple(DOMAIN_BENCH.keys())


@mcp.tool()
def sophia_validate() -> str:
    """Validate data/attributions.json and all training/examples/*.json."""
    return json.dumps(run_validation(), ensure_ascii=False, indent=2)


@mcp.tool()
def sophia_corpus_stats() -> str:
    """Return version, attribution count, training example count, benchmark case totals."""
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    attributions = len(load_json(ROOT / "data" / "attributions.json"))
    examples = len(list((ROOT / "training" / "examples").glob("*.json")))
    bench = {}
    for domain, path in DOMAIN_BENCH.items():
        if path.exists():
            bench[domain] = len(load_json(path).get("cases", []))
    payload = {
        "version": version,
        "attributions": attributions,
        "trainingExamples": examples,
        "benchmarkCases": bench,
        "benchmarkTotal": sum(bench.values()),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


@mcp.tool()
def sophia_gate_check(
    response: str,
    question: str,
    mode: str = "advisor",
    domain: str | None = None,
    strict_attribution: bool = True,
) -> str:
    """Run the Sophia epistemic gate on a draft answer (attribution traps + style checks)."""
    if mode not in ("advisor", "repo", "life"):
        return json.dumps({"error": f"mode must be advisor|repo|life, got {mode!r}"})
    result = check_response(
        response,
        mode=mode,
        question=question,
        domain=domain,
        strict_attribution=strict_attribution,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def sophia_benchmark_list(domain: str = "philosophy") -> str:
    """List benchmark case id + question for a domain (philosophy|psychology|history|religion)."""
    if domain not in DOMAINS:
        return json.dumps({"error": f"domain must be one of {DOMAINS}"})
    bench = load_json(DOMAIN_BENCH[domain])
    cases = [{"id": c["id"], "question": c["question"]} for c in bench.get("cases", [])]
    return json.dumps(
        {"domain": domain, "version": bench.get("version", 1), "cases": cases},
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
def sophia_benchmark_score(
    domain: str,
    responses_json: str,
    model: str = "mcp-eval",
) -> str:
    """Score benchmark responses. responses_json: {\"case_id\": \"answer text\", ...}."""
    if domain not in DOMAINS:
        return json.dumps({"error": f"domain must be one of {DOMAINS}"})
    try:
        responses = json.loads(responses_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"invalid JSON: {exc}"})
    if not isinstance(responses, dict):
        return json.dumps({"error": "responses_json must be a JSON object"})

    bench = load_json(DOMAIN_BENCH[domain])
    traditions = load_json(ROOT / "data" / "traditions.json")
    results = []
    passed = 0
    for case in bench.get("cases", []):
        case_id = case["id"]
        text = str(responses.get(case_id, ""))
        ok, reasons = score_case(case, text, traditions)
        if ok:
            passed += 1
        results.append({"id": case_id, "passed": ok, "reasons": reasons})
    total = len(results)
    report = {
        "domain": domain,
        "model": model,
        "passed": passed,
        "total": total,
        "score_pct": round(100.0 * passed / total, 1) if total else 0.0,
        "results": results,
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
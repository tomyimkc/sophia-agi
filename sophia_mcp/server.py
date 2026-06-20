#!/usr/bin/env python3
"""Sophia AGI MCP server — validate, gate, benchmark, corpus lookup.

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

from sophia_mcp.tools_impl import (  # noqa: E402
    benchmark_list,
    benchmark_score,
    corpus_stats,
    dumps,
    export_corpus,
    gate_check,
    get_attribution,
    get_record,
    list_disputes,
    read_dispute,
    rubric_review,
    sector_council,
    validate_corpus,
    web_evidence_search,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install MCP deps: pip install -r requirements-mcp.txt") from exc

mcp = FastMCP(
    "sophia-agi",
    instructions=(
        "Sophia AGI provenance tools: validate corpus, epistemic gate, benchmarks, "
        "attribution lookup, dispute notes, corpus export, and law/financial/economy "
        "sector councils."
    ),
)


@mcp.tool()
def sophia_validate() -> str:
    """Validate data/attributions.json and all training/examples/*.json."""
    return dumps(validate_corpus())


@mcp.tool()
def sophia_corpus_stats() -> str:
    """Return version, attribution count, training example count, benchmark case totals."""
    return dumps(corpus_stats())


@mcp.tool()
def sophia_export_corpus() -> str:
    """Export training/examples/*.json to training/corpus.jsonl."""
    return dumps(export_corpus())


@mcp.tool()
def sophia_gate_check(
    response: str,
    question: str,
    mode: str = "advisor",
    domain: str | None = None,
    strict_attribution: bool = True,
) -> str:
    """Run the Sophia epistemic gate on a draft answer (attribution traps + style checks)."""
    return dumps(gate_check(response, question, mode=mode, domain=domain, strict_attribution=strict_attribution))


@mcp.tool()
def sophia_benchmark_list(domain: str = "philosophy") -> str:
    """List benchmark case id + question for a domain (philosophy|psychology|history|religion)."""
    return dumps(benchmark_list(domain))


@mcp.tool()
def sophia_benchmark_score(
    domain: str,
    responses_json: str,
    model: str = "mcp-eval",
) -> str:
    """Score benchmark responses. responses_json: {\"case_id\": \"answer text\", ...}."""
    try:
        responses = json.loads(responses_json)
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid JSON: {exc}"})
    if not isinstance(responses, dict):
        return dumps({"error": "responses_json must be a JSON object"})
    return dumps(benchmark_score(domain, responses, model=model))


@mcp.tool()
def sophia_get_attribution(text_id: str) -> str:
    """Lookup a philosophy attribution record from data/attributions.json by textId."""
    return dumps(get_attribution(text_id))


@mcp.tool()
def sophia_get_record(domain: str, record_id: str) -> str:
    """Lookup domain record: philosophy|psychology|history|religion + record id."""
    return dumps(get_record(domain, record_id))


@mcp.tool()
def sophia_list_disputes() -> str:
    """List dispute note slugs in docs/04-Disputes/."""
    return dumps(list_disputes())


@mcp.tool()
def sophia_read_dispute(slug: str) -> str:
    """Read a dispute markdown file by slug (e.g. Laozi-Dao-De-Jing-Attribution)."""
    return dumps(read_dispute(slug))


@mcp.tool()
def sophia_web_evidence_search(
    query: str,
    online: bool = False,
    provider: str = "off",
    top_k: int = 5,
    local_top_k: int = 3,
) -> str:
    """Search Sophia local RAG plus optional Brave/Tavily/SerpAPI web evidence."""
    return dumps(web_evidence_search(query, online=online, provider=provider, top_k=top_k, local_top_k=local_top_k))


@mcp.tool()
def sophia_sector_council(
    council_id: str,
    query: str,
    materials_json: str = "[]",
) -> str:
    """Convene a Sophia sector council for a query: council_id = law|financial|economy|auto.

    Seats source-inspired specialists plus standing guardians (citation/numbers
    audit, ethics/equity, plain-language, human-review gate) and returns the
    council prompt, seated seats, decision contract, and human-authority boundary.
    Decision support only — not professional legal/financial advice.
    """
    try:
        materials = json.loads(materials_json)
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid materials JSON: {exc}"})
    if not isinstance(materials, list):
        return dumps({"error": "materials_json must be a JSON array"})
    return dumps(sector_council(council_id, query, materials=materials))


@mcp.tool()
def sophia_rubric_review(
    question: str,
    response: str,
    domain: str = "philosophy",
    must_include_json: str = "[]",
    must_avoid_json: str = "[]",
) -> str:
    """Review a draft against required/forbidden rubric items plus Sophia gate checks."""
    try:
        must_include = json.loads(must_include_json)
        must_avoid = json.loads(must_avoid_json)
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid rubric JSON: {exc}"})
    if not isinstance(must_include, list) or not isinstance(must_avoid, list):
        return dumps({"error": "must_include_json and must_avoid_json must be JSON arrays"})
    return dumps(
        rubric_review(
            question,
            response,
            domain=domain,
            must_include=must_include,
            must_avoid=must_avoid,
        )
    )


if __name__ == "__main__":
    mcp.run()

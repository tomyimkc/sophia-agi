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
    validate_corpus,
)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:
    raise SystemExit("Install MCP deps: pip install -r requirements-mcp.txt") from exc

mcp = FastMCP(
    "sophia-agi",
    instructions=(
        "Sophia AGI provenance tools: validate corpus, epistemic gate, benchmarks, "
        "attribution lookup, dispute notes, and corpus export."
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


if __name__ == "__main__":
    mcp.run()
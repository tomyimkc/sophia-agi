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
    belief,
    benchmark_list,
    benchmark_score,
    check_claim,
    corpus_stats,
    council_deliberate,
    counterfactual,
    dumps,
    export_corpus,
    gate_check,
    get_attribution,
    get_record,
    list_disputes,
    openclaw_infer,
    read_dispute,
    retract,
    revise,
    rubric_review,
    sector_council,
    validate_corpus,
    web_evidence_search,
    wiki_contradictions,
    wiki_read,
    wiki_search,
    wiki_upsert,
    wiki_validate_tool,
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
def sophia_check_claim(text: str) -> str:
    """Mode-free source-discipline check: is an attribution forbidden by Sophia's
    'don't merge lineages' rule? Returns {passed, reasons, violations}. Read-only."""
    return dumps(check_claim(text))


@mcp.tool()
def sophia_belief(entity: str) -> str:
    """Belief-graph lookup for an entity (page id/alias): effectiveConfidenceRank
    (min over the derivesFrom chain), declared confidence, attribution, and a
    confidenceLaundered flag. Read-only."""
    return dumps(belief(entity))


@mcp.tool()
def sophia_counterfactual(source: str, query: str | None = None) -> str:
    """Counterfactual belief-graph query: "what would I conclude if this source
    were removed?" Returns affected claims with grounding before/after and a
    supportLost list (claims that lose their only ground — fail-closed). Optional
    query isolates one entity's before/after belief. Read-only, non-destructive."""
    return dumps(counterfactual(source, query=query))


@mcp.tool()
def sophia_retract(target: str, reason: str, by: str = "system") -> str:
    """Retract a claim: a named, auditable decision that computes downstream
    impact (which claims lose support) and returns an append-only audit entry.
    Non-destructive — no page is deleted. Read-only over the live graph."""
    return dumps(retract(target, reason, by=by))


@mcp.tool()
def sophia_revise(targets: list[str], reason: str = "(unspecified)", by: str = "system") -> str:
    """Belief revision: apply one or more retractions and propagate the support
    cascade transitively. Returns retracted ids, the cascade of claims that lose
    support, the abstain set (what a gate must now refuse), and an audit log.
    Non-destructive, read-only over the live graph."""
    return dumps(revise(targets, reason, by=by))


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
def sophia_council_deliberate(
    query: str,
    model: str = "mock",
    max_seats: int = 4,
    gate: bool = True,
) -> str:
    """Deliberate a query across a council: a focused pass per seat, a per-seat gate,
    then synthesis (map-reduce). The small-LLM uplift — narrow gated passes beat one
    shallow pass. `model` is a Sophia model spec (mock|ollama:..|openrouter:..|..).
    Returns per-seat answers, which seats were gated out, and the synthesised
    decision. Decision support only — not professional advice.
    """
    return dumps(council_deliberate(query, model=model, max_seats=max_seats, gate=gate))


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


@mcp.tool()
def sophia_wiki_read(page_id: str) -> str:
    """Read an OKF wiki page (frontmatter + body) by id from the provenance wiki."""
    return dumps(wiki_read(page_id))


@mcp.tool()
def sophia_wiki_search(query: str, top_k: int = 8) -> str:
    """Keyword-search the OKF wiki; returns matching page ids/types (not raw chunks)."""
    return dumps(wiki_search(query, top_k=top_k))


@mcp.tool()
def sophia_wiki_contradictions() -> str:
    """Structural contradiction ledger over the wiki: lineage merges, cycles, laundering."""
    return dumps(wiki_contradictions())


@mcp.tool()
def sophia_wiki_validate() -> str:
    """Validate the OKF wiki: schema, link integrity, contradictions, data drift."""
    return dumps(wiki_validate_tool())


@mcp.tool()
def sophia_wiki_upsert(page_id: str, frontmatter_json: str = "{}", body: str = "", tier: str = "draft") -> str:
    """Create/update an agent-owned wiki page (gated + audited).

    Mutating: needs SOPHIA_MCP_APPROVE_WRITES=1. The write only lands if it passes
    the source-discipline gate (schema-valid + no forbidden attribution/lineage merge).
    """
    return dumps(wiki_upsert(page_id, frontmatter_json=frontmatter_json, body=body, tier=tier))


@mcp.tool()
def sophia_openclaw_infer(prompt: str, model: str = "xai/grok-4.3") -> str:
    """Read-only text inference via the local OpenClaw gateway CLI (risk=low, audited).

    OpenClaw owns provider auth/fallback; `model` is its <provider>/<model> route. This is
    pure inference and NOT a knowledge-write path — OpenClaw output only enters the wiki
    via sophia_wiki_upsert and the source-discipline gate (no lineage merge can be written).
    """
    return dumps(openclaw_infer(model=model, prompt=prompt))


if __name__ == "__main__":
    mcp.run()

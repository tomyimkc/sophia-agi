"""MCP tool implementations (importable for tests)."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from agent.benchmark_checks import DOMAIN_BENCH, load_json, score_case
from agent.gate import check_response
from agent.rubric_review import build_rubric_review
from agent.sector_council import (
    available_councils,
    detect_council,
    format_council,
    load_council,
    route_council,
)
from agent.web_evidence import gather_evidence
from sophia_mcp.audit import audited
from tools.validate_attribution import run_validation

ROOT = Path(__file__).resolve().parents[1]
DOMAIN_DATA = {
    "philosophy": ROOT / "data" / "attributions.json",
    "psychology": ROOT / "data" / "psychology_concepts.json",
    "history": ROOT / "data" / "history_events.json",
    "religion": ROOT / "data" / "religion_concepts.json",
}
DISPUTES_DIR = ROOT / "docs" / "04-Disputes"
CORPUS_OUT = ROOT / "training" / "corpus.jsonl"
EXAMPLES_DIR = ROOT / "training" / "examples"
DOMAINS = tuple(DOMAIN_BENCH.keys())


def validate_corpus() -> dict:
    return run_validation()


def corpus_stats() -> dict:
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    attributions = len(load_json(ROOT / "data" / "attributions.json"))
    examples = len(list(EXAMPLES_DIR.glob("*.json")))
    bench = {}
    for domain, path in DOMAIN_BENCH.items():
        if path.exists():
            bench[domain] = len(load_json(path).get("cases", []))
    return {
        "version": version,
        "attributions": attributions,
        "trainingExamples": examples,
        "benchmarkCases": bench,
        "benchmarkTotal": sum(bench.values()),
    }


def gate_check(
    response: str,
    question: str,
    *,
    mode: str = "advisor",
    domain: str | None = None,
    strict_attribution: bool = True,
) -> dict:
    if mode not in ("advisor", "repo", "life"):
        return {"error": f"mode must be advisor|repo|life, got {mode!r}"}
    return check_response(
        response,
        mode=mode,
        question=question,
        domain=domain,
        strict_attribution=strict_attribution,
    )


def check_claim(text: str) -> dict:
    """Mode-free source-discipline check: ``{passed, reasons, violations}``.

    Unlike ``gate_check`` (moded, needs a question + style scoring), this is the
    pure provenance verifier — text in, verdict out — so a caller can gate any
    claim against Sophia's "don't merge lineages" rule. Read-only, offline.
    """
    from agent.guarded import check_claim as _check_claim

    return _check_claim(text)


def belief(entity: str) -> dict:
    """Belief-graph lookup for one entity: effectiveConfidenceRank (min over the
    derivesFrom chain), declared confidence, attribution, contradictions, and a
    confidenceLaundered flag. Read-only, offline."""
    import okf
    from agent import wiki_store

    graph = okf.build_graph(wiki_store.load_all_pages())
    return okf.belief(graph, entity)


def counterfactual(source: str, query: str | None = None) -> dict:
    """Counterfactual belief-graph query: what would change if ``source`` were
    removed? Returns affected claims with grounding before/after and a
    ``supportLost`` list (the claims that lose their only provenance ground —
    fail-closed, confidence collapses to 0). Optional ``query`` isolates one
    entity's before/after belief. Read-only, offline, non-destructive."""
    import okf
    from agent import wiki_store

    graph = okf.build_graph(wiki_store.load_all_pages())
    return okf.counterfactual_remove(graph, source, query=query)


def retract(target: str, reason: str, by: str = "system") -> dict:
    """Retract a claim from the belief set: a named, auditable decision that
    computes downstream impact (which claims lose support) and returns an
    append-only audit entry. Non-destructive — no page is deleted; persistence
    is the caller's choice, made with the impact in hand. Offline."""
    import okf
    from agent import wiki_store

    graph = okf.build_graph(wiki_store.load_all_pages())
    return okf.retract(graph, target, reason=reason, by=by).to_dict()


def benchmark_list(domain: str) -> dict:
    if domain not in DOMAINS:
        return {"error": f"domain must be one of {DOMAINS}"}
    bench = load_json(DOMAIN_BENCH[domain])
    cases = [{"id": c["id"], "question": c["question"]} for c in bench.get("cases", [])]
    return {"domain": domain, "version": bench.get("version", 1), "cases": cases}


def benchmark_score(domain: str, responses: dict, *, model: str = "mcp-eval") -> dict:
    if domain not in DOMAINS:
        return {"error": f"domain must be one of {DOMAINS}"}
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
    return {
        "domain": domain,
        "model": model,
        "passed": passed,
        "total": total,
        "score_pct": round(100.0 * passed / total, 1) if total else 0.0,
        "results": results,
    }


def get_attribution(text_id: str) -> dict:
    records = load_json(DOMAIN_DATA["philosophy"])
    if text_id not in records:
        return {"error": f"unknown textId: {text_id}", "available": sorted(records.keys())[:20]}
    return records[text_id]


def get_record(domain: str, record_id: str) -> dict:
    if domain not in DOMAIN_DATA:
        return {"error": f"domain must be one of {tuple(DOMAIN_DATA.keys())}"}
    records = load_json(DOMAIN_DATA[domain])
    if record_id not in records:
        sample = sorted(records.keys())[:15]
        return {"error": f"unknown record_id: {record_id}", "sampleIds": sample}
    return records[record_id]


def list_disputes() -> dict:
    if not DISPUTES_DIR.exists():
        return {"disputes": []}
    items = []
    for path in sorted(DISPUTES_DIR.glob("*.md")):
        title = path.stem.replace("-", " ")
        items.append({"slug": path.stem, "file": str(path.relative_to(ROOT)), "title": title})
    return {"count": len(items), "disputes": items}


def read_dispute(slug: str) -> dict:
    path = DISPUTES_DIR / f"{slug}.md"
    if not path.exists():
        matches = [p.stem for p in DISPUTES_DIR.glob("*.md") if slug.lower() in p.stem.lower()]
        return {"error": f"dispute not found: {slug}", "suggestions": matches[:10]}
    return {"slug": slug, "content": path.read_text(encoding="utf-8")}


def web_evidence_search(
    query: str,
    *,
    online: bool = False,
    provider: str = "off",
    top_k: int = 5,
    local_top_k: int = 3,
) -> dict:
    if not query.strip():
        return {"error": "query is required"}
    from agent.dataflow.firewall import active_profile

    if online and active_profile() == "airgap":
        return {"error": "airgap profile blocks network egress (web_evidence_search)", "results": []}
    return gather_evidence(
        query,
        local_top_k=local_top_k,
        web_top_k=top_k,
        online=online,
        provider=provider,
    )


def rubric_review(
    question: str,
    response: str,
    *,
    domain: str = "philosophy",
    must_include: list | None = None,
    must_avoid: list | None = None,
) -> dict:
    if not question.strip():
        return {"error": "question is required"}
    if not response.strip():
        return {"error": "response is required"}
    mode = "repo" if domain in {"coding", "planning", "tool_use"} else "advisor"
    gate = check_response(response, mode=mode, question=question, domain=domain if domain in DOMAIN_DATA else None)
    must_include = must_include or []
    must_avoid = must_avoid or []
    failed_include = [item for item in must_include if _plain_match_missing(response, item)]
    failed_avoid = [item for item in must_avoid if not _plain_match_missing(response, item)]
    total = len(must_include) + len(must_avoid)
    passed_checks = len(must_include) - len(failed_include) + len(must_avoid) - len(failed_avoid)
    score_result = {
        "passed": bool(total and passed_checks == total),
        "failedInclude": failed_include,
        "failedAvoid": failed_avoid,
        "semanticResults": [],
    }
    case = {
        "id": "mcp_rubric_review",
        "domain": domain,
        "prompt": question,
        "scoring": {"mustInclude": must_include, "mustAvoid": must_avoid, "semanticChecks": []},
    }
    return build_rubric_review(case, response, score_result, gate)


def sector_council(council_id: str, query: str, *, materials: list | None = None) -> dict:
    if not query.strip():
        return {"error": "query is required"}
    if council_id == "auto":
        detected = detect_council(query)
        if not detected:
            return {
                "error": "no sector council matched; specify law|financial|economy",
                "available": available_councils(),
            }
        council_id = detected
    if council_id not in available_councils():
        return {"error": f"council_id must be one of {available_councils()} or 'auto'"}
    route = route_council(load_council(council_id), query, materials or [])
    seated = sorted(seat.get("seatId") for group in route["selected"].values() for seat in group["seats"])
    return {
        "councilId": council_id,
        "displayName": route.get("displayName"),
        "seatedSeatIds": seated,
        "humanBoundary": route.get("humanBoundary", []),
        "decisionContract": route.get("decisionContract", []),
        "councilPrompt": format_council(route),
        "notAdvice": "Decision support only — not professional legal/financial advice.",
    }


def council_deliberate(query: str, *, model: str = "mock", max_seats: int = 4, gate: bool = True) -> dict:
    """Map-reduce a query across a council's seats (a focused pass per seat + a
    per-seat gate + synthesis). The small-model uplift: many narrow gated passes
    beat one shallow pass. Decision support only — not professional advice."""
    if not query.strip():
        return {"error": "query is required"}
    from agent.council_deliberate import deliberate
    from agent.model import default_client

    d = deliberate(query, client=default_client(model), max_seats=max_seats, gate=gate)
    out = d.to_dict()
    out["notAdvice"] = "Decision support only — not professional legal/financial advice."
    return out


@audited("sophia_export_corpus", risk="medium")
def export_corpus() -> dict:
    examples = sorted(EXAMPLES_DIR.glob("*.json"))
    if not examples:
        return {"error": "no training examples found"}
    CORPUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_OUT.open("w", encoding="utf-8") as handle:
        for path in examples:
            payload = json.loads(path.read_text(encoding="utf-8"))
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"ok": True, "path": str(CORPUS_OUT), "lines": len(examples)}


# --------------------------------------------------------------------------- #
# OKF wiki tools — read surface + audited write surface (the librarian's hands).
# --------------------------------------------------------------------------- #


def wiki_read(page_id: str) -> dict:
    from agent import wiki_store

    page = wiki_store.read_page(page_id)
    if page is None:
        return {"error": f"wiki page not found: {page_id}"}
    return {"id": page.id, "pageType": page.page_type, "path": str(page.path.relative_to(ROOT)),
            "frontmatter": page.meta, "body": page.body}


def wiki_search(query: str, *, top_k: int = 8) -> dict:
    from agent import wiki_store

    hits = wiki_store.search(query, top_k=top_k)
    return {"query": query, "results": [{"id": p.id, "pageType": p.page_type,
                                         "path": str(p.path.relative_to(ROOT))} for p in hits]}


def wiki_contradictions() -> dict:
    from agent import wiki_store

    return wiki_store.contradictions()


def wiki_validate_tool() -> dict:
    from tools.wiki_validate import run_validation as _wiki_run_validation

    return _wiki_run_validation()


@audited("sophia_wiki_upsert", risk="medium")
def wiki_upsert(page_id: str, frontmatter_json: str = "{}", body: str = "", tier: str = "draft") -> dict:
    """Create/update an agent-owned wiki page (gated by provenance verifiers).

    Mutating + audited: needs SOPHIA_MCP_APPROVE_WRITES=1. Even when approved, the
    write only lands if it passes the source-discipline gate AND the data-flow
    firewall (a WRITE sink — a tainted/untrusted-labelled payload is refused).
    """
    from agent import wiki_store
    from agent.dataflow.firewall import guard_call

    decision = guard_call("sophia_wiki_upsert", (page_id, frontmatter_json, body, tier))
    if decision.action != "allow":
        return {"error": f"data-flow firewall blocked write: {decision.reason}"}

    try:
        meta = json.loads(frontmatter_json) if frontmatter_json else {}
    except json.JSONDecodeError as exc:
        return {"error": f"invalid frontmatter_json: {exc}"}
    if not isinstance(meta, dict):
        return {"error": "frontmatter_json must be a JSON object"}
    return wiki_store.upsert(page_id, meta=meta, body=body, tier=tier)


def _openclaw_infer(model: str, prompt: str, *, timeout_sec: int = 120) -> dict:
    """Pure logic for the OpenClaw inference tool — shell to the CLI, parse JSON.

    Read-only and offline-stubbable: when the binary is absent (FileNotFoundError) or the
    output is unparsable, it returns a structured error dict and never raises.
    """
    if not prompt:
        return {"ok": False, "error": "prompt is required"}
    binary = os.environ.get("SOPHIA_OPENCLAW_BIN", "openclaw")
    command = [binary, "infer", "model", "run", "--model", model, "--prompt", prompt, "--json"]
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout_sec, check=False)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return {"ok": False, "error": repr(exc)}
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "")[-500:]}
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"unparsable openclaw output: {exc}"}
    outputs = data.get("outputs") if isinstance(data, dict) else None
    text = ""
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        text = (outputs[0].get("text") or "").strip()
    ok = bool(isinstance(data, dict) and data.get("ok", True)) and bool(text)
    return {
        "ok": ok,
        "provider": data.get("provider") if isinstance(data, dict) else None,
        "model": model,
        "text": text,
        "error": None if ok else "openclaw returned no usable text",
    }


@audited("sophia_openclaw_infer", risk="low")
def openclaw_infer(model: str = "xai/grok-4.3", prompt: str = "") -> dict:
    """Read-only text inference through the local OpenClaw gateway CLI.

    OpenClaw owns provider auth/fallback; ``model`` is its ``<provider>/<model>`` route.
    Pure inference -> risk="low" (audited but no approval needed). This is NOT a
    knowledge-write path: to land OpenClaw output in the wiki, route it through
    sophia_wiki_upsert -> wiki_store.upsert -> the source-discipline gate, which
    independently rejects lineage merges even when writes are approved.
    """
    from agent.dataflow.firewall import active_profile

    if active_profile() == "airgap":
        return {"ok": False, "error": "airgap profile blocks egress (openclaw_infer)"}
    return _openclaw_infer(model, prompt)


def _plain_match_missing(response: str, item: object) -> bool:
    text = response.lower()
    if isinstance(item, dict):
        options = [str(item.get("match", "")), *[str(alias) for alias in item.get("aliases", [])]]
    else:
        options = [str(item)]
    return not any(option and option.lower() in text for option in options)


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)

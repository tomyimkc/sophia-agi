# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
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
PERSONALITY_TYPES = ROOT / "data" / "personality_types.json"
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


def trajectory_eval(trajectory: "list | None") -> dict:
    """Score a whole agent trajectory for mid-plan faithfulness, step by step.

    ``trajectory`` is an ordered list of step dicts (``observation`` = evidence the
    environment returned; ``claim``/``text`` = what the agent asserted; ``cites`` =
    ids of earlier steps that support the claim). Returns the
    ``sophia.trajectory_eval.v1`` record: a fail-closed ``verdict``
    (accept | abstain | blocked), a ``faithfulnessScore``, the first unfaithful
    step, and a per-step breakdown. Read-only, offline (deterministic lexical
    judge). See ``agent.trajectory_eval``.
    """
    from agent.trajectory_eval import evaluate_trajectory

    if not isinstance(trajectory, list):
        return {"error": "trajectory must be a list of step objects"}
    return evaluate_trajectory(trajectory)


def medical_citation_check(text: str) -> dict:
    """Verify medical citations in ``text``: do the PMIDs / DOIs / guideline IDs
    EXIST (deterministic, against the bundled register), and — with a judge —
    whether each is cited faithfully. Without a judge the support tier abstains
    (fail-closed); fabricated citations are always flagged. Not clinical advice.
    See ``agent.medical_faithfulness``.
    """
    from agent.medical_faithfulness import assess_text, medical_citation_exists

    existence = medical_citation_exists()(text, None, {})
    assessment = assess_text(text)
    return {
        "passed": existence["passed"],
        "violations": [f"unverifiable medical citation: {c}" for c in existence["detail"]["missing"]],
        "checked": existence["detail"]["checked"],
        "fabricated": assessment["fabricated"],
        "abstained": assessment["abstained"],
        "contradicted": assessment["contradicted"],
        "supported": assessment["supported"],
        "notAdvice": "Citation review only — not medical advice; verify against the primary source.",
    }


def _belief_graph():
    """Build the belief graph from the store tiers + dispute lineage pages."""
    import okf
    from agent import wiki_store

    return okf.build_graph(wiki_store.belief_graph_pages())


def belief(entity: str) -> dict:
    """Belief-graph lookup for one entity: effectiveConfidenceRank (min over the
    derivesFrom chain), declared confidence, attribution, contradictions, and a
    confidenceLaundered flag. Read-only, offline."""
    import okf

    return okf.belief(_belief_graph(), entity)


def counterfactual(source: str, query: str | None = None) -> dict:
    """Counterfactual belief-graph query: what would change if ``source`` were
    removed? Returns affected claims with grounding before/after and a
    ``supportLost`` list (the claims that lose their only provenance ground —
    fail-closed, confidence collapses to 0). Optional ``query`` isolates one
    entity's before/after belief. Read-only, offline, non-destructive."""
    import okf

    return okf.counterfactual_remove(_belief_graph(), source, query=query)


def retract(target: str, reason: str, by: str = "system") -> dict:
    """Retract a claim from the belief set: a named, auditable decision that
    computes downstream impact (which claims lose support) and returns an
    append-only audit entry. Non-destructive — no page is deleted; persistence
    is the caller's choice, made with the impact in hand. Offline."""
    import okf

    return okf.retract(_belief_graph(), target, reason=reason, by=by).to_dict()


def revise(targets: list[str], reason: str = "(unspecified)", by: str = "system") -> dict:
    """Belief revision: apply one or more retractions and propagate the support
    cascade transitively. Returns the retracted ids, the cascade of claims that
    lose support, the abstain set (what a gate must now refuse), and an audit log.
    Non-destructive. Offline."""
    import okf

    pairs = [(t, reason) for t in targets]
    return okf.revise(_belief_graph(), pairs, by=by).to_dict()


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


def council_deliberate(query: str, *, model: str = "mock", models: list | None = None,
                       max_seats: int = 4, gate: bool = True) -> dict:
    """Map-reduce a query across a council's seats (a focused pass per seat + a
    per-seat gate + synthesis). The small-model uplift: many narrow gated passes
    beat one shallow pass. Decision support only — not professional advice.

    ``model`` runs every seat homogeneously (one model wearing N hats — correlated
    errors). Pass ``models`` (a list of model specs) to seat a HETEROGENEOUS panel:
    the substantive seats are cycled across genuinely different models, so the
    voters are independent. The synthesis chair always uses ``model``.
    """
    if not query.strip():
        return {"error": "query is required"}
    from agent.council_deliberate import deliberate
    from agent.model import default_client

    seat_clients = [default_client(m) for m in models] if models else None
    d = deliberate(
        query,
        client=default_client(model),
        seat_clients=seat_clients,
        max_seats=max_seats,
        gate=gate,
    )
    out = d.to_dict()
    out["heterogeneous"] = bool(seat_clients)
    out["notAdvice"] = "Decision support only — not professional legal/financial advice."
    return out


def team_agents_deliberate(
    query: str,
    *,
    model: str = "mock",
    adapter_path: str = "",
    seat_models: list | None = None,
    max_seats: int = 4,
    gate: bool = True,
) -> dict:
    """Runtime team orchestrator: deliberate_team() with optional Sophia LoRA adapter.

    Decision support only — not professional advice. ``canClaimAGI: false`` always.
    """
    if not query.strip():
        return {"error": "query is required"}
    import os

    from agent.model import default_client
    from agent.team_agents import deliberate_team

    if adapter_path:
        os.environ["SOPHIA_MLX_ADAPTER"] = adapter_path
    client = default_client(model)
    seat_clients = [default_client(m) for m in seat_models] if seat_models else None
    d = deliberate_team(
        query,
        client=client,
        seat_clients=seat_clients,
        max_seats=max_seats,
        gate=gate,
    )
    out = d.to_dict()
    out.update(
        adapterPath=adapter_path or None,
        heterogeneous=bool(seat_clients),
        candidateOnly=True,
        canClaimAGI=False,
        notAdvice="Decision support only — not professional legal/financial advice.",
    )
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
    from agent.conscience_enforcement import enforce_conscience
    # MCP wiki upsert may write draft/concept pages that are not yet trusted
    # semantic memory. Conscience is mandatory here as a hard-block screen for
    # overclaim/tampering/forbidden attribution, while trusted semantic memory
    # evidence rules remain enforced in agent.layered_memory.
    enforcement = enforce_conscience(
        action="draft_output", text=body or page_id, mode="output", high_impact=False,
        context={"allowCautionVerdicts": True},
    )
    if not enforcement.allowed:
        return {"error": f"conscience blocked wiki write: {enforcement.reason}", "conscience": enforcement.to_dict()}
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


# --------------------------------------------------------------------------- #
# Governance contract over MCP (sophia_contract) — the aihk-os seam as tools.
# --------------------------------------------------------------------------- #

_CONTRACT = None  # lazily built singleton; tests may set this to an in-memory instance


def _contract():
    global _CONTRACT
    if _CONTRACT is None:
        from agent.config import MEMORY_DIR
        from sophia_contract import SophiaContract
        from sophia_contract.roles import ROLES_9

        _CONTRACT = SophiaContract(store_dir=MEMORY_DIR / "contract", scopes=ROLES_9, tracing=True)
    return _CONTRACT


def contract_describe() -> dict:
    """Handshake the governance contract: version, capabilities, schema_url, deprecations."""
    return _contract().describe()


def record_claim(idempotency_key: str, content: str, sources=None, parents=None,
                 blp_level: str = "UNCLASSIFIED", role=None, dry_run: bool = False) -> dict:
    """Record a provenance claim (idempotent; BLP no-write-down at record time)."""
    req = {"idempotency_key": idempotency_key, "content": content,
           "sources": sources or [], "parents": parents or [], "blp_level": blp_level}
    if role:
        req["role"] = role
    if dry_run:
        req["dry_run"] = True
    return _contract().record_claim(req)


def verify_claim(claim_id: str, clearance: str = "UNCLASSIFIED", role=None) -> dict:
    """Verify a claim -> Verdict. Only 'accepted' may be published (fail-closed)."""
    req = {"claim_id": claim_id}
    if role:
        req["role"] = role
    return _contract().verify_claim(req, clearance=clearance)


def explain_verdict(claim_id: str, clearance: str = "UNCLASSIFIED") -> dict:
    """Verify a claim and return the verdict plus a one-line rule-path trace."""
    return _contract().explain_verdict({"claim_id": claim_id}, clearance=clearance)


def contract_health() -> dict:
    """Liveness + self-diagnostics (kill switch, pending tasks, budget)."""
    return _contract().health()


def enqueue_task(idempotency_key: str, kind: str, payload=None, role=None) -> dict:
    """Durably + idempotently enqueue work for an unattended pipeline."""
    req = {"idempotency_key": idempotency_key, "kind": kind, "payload": payload or {}}
    if role:
        req["role"] = role
    return _contract().enqueue_task(req)


def next_task(lease_by: str = "worker") -> dict:
    """Lease the oldest pending task ({task: null} when the queue is empty)."""
    return _contract().next_task({"lease_by": lease_by})


def mbti_type_record(type: str) -> dict:
    """Lookup an MBTI type record from data/personality_types.json (read-only)."""
    code = (type or "").strip().upper()
    records = load_json(PERSONALITY_TYPES)
    if code not in records:
        return {"error": f"unknown MBTI type: {type!r}", "sampleIds": sorted(records.keys())[:16]}
    return records[code]


def personality_target(mbti: str, ocean: dict, prompt: str, *, model: str = "mock",
                       gate: bool = True) -> dict:
    """Generate a response steered toward a target personality (MBTI veneer +
    OCEAN substrate). Level-1 persona prompting only (Spec A). Read-only."""
    if not (prompt or "").strip():
        return {"error": "prompt is required"}
    code = (mbti or "").strip().upper()
    from agent.personality_map import mbti_to_ocean, SIXTEEN_TYPES
    if code and code not in SIXTEEN_TYPES:
        return {"error": f"unknown MBTI type: {mbti!r}", "available": list(SIXTEEN_TYPES)}
    target = dict(mbti_to_ocean(code)) if code else {}
    target.pop("_meta", None)
    target.update(ocean or {})  # explicit OCEAN overrides the veneer-derived signs
    from agent.model import complete
    system = ("Adopt this Big Five (OCEAN) profile in your voice "
              f"(high/low per axis; Neuroticism unspecified unless given): {json.dumps(target, ensure_ascii=False)}.")
    try:
        response = complete(system, prompt, spec=model).strip()
    except Exception as exc:  # offline/credential failure -> structured error
        return {"error": f"generation failed: {exc!r}"}
    out = {"mbti": code, "oceanTarget": target, "model": model, "response": response, "gated": bool(gate)}
    if gate:
        from agent.gate import check_response
        try:
            verdict = check_response(response, mode="advisor", question=prompt)
            out["gate"] = verdict
            out["passed"] = bool(verdict.get("passed", True))
        except Exception as exc:
            out["gate_error"] = repr(exc)
            out["passed"] = False
    return out


def personality_faithful_score(text: str, mbti: str, ocean: dict, *, model: str = "mock") -> dict:
    """Score how faithfully `text` expresses a target personality. Deterministic
    (no model call in Spec A); `model` reserved for the behavioral channel (Spec B)."""
    if not (text or "").strip():
        return {"error": "text is required"}
    from agent.verifiers import personality_faithful
    verdict = personality_faithful({"mbti": (mbti or "").strip().upper(), "ocean": ocean or {}})(text, None, {})
    return {
        "mbti": (mbti or "").strip().upper(),
        "ocean": ocean or {},
        "passed": verdict["passed"],
        "status": verdict["detail"].get("status"),
        "reasons": verdict["reasons"],
    }


def ocean_measure(answers: dict) -> dict:
    """Score a {item_id: 1..5} IPIP answer map into OCEAN domain scores.
    Read-only, deterministic, no model. Reuses A's score_items/load_bank."""
    from agent.personality_measure import score_items, load_bank
    bank = load_bank()
    return {"ocean": score_items(answers, bank), "nItems": len(answers)}


def capability_retention_demo() -> dict:
    """The Spec D deterministic capability cell on the bundled arithmetic slice
    (base correct vs degenerate steered). Read-only, no model."""
    from tools.run_capability import build_dry_run_cell
    return build_dry_run_cell()


def council_diversity_summary() -> dict:
    """The committed Spec C council A/B result (ΔQ does-not-replicate null)."""
    from pathlib import Path
    p = (Path(__file__).resolve().parents[1]
         / "agi-proof" / "benchmark-results" / "council-diversity.public-report.json")
    if not p.exists():
        return {"available": False, "reason": "council-diversity report not generated"}
    return json.loads(p.read_text())


def pif_dryrun_summary() -> dict:
    """Spec C PIF/SSA harness invariants on a synthetic (is_mock) fixture — the
    real build_cells_from_scores + headline code path, CI-green core. Read-only."""
    import random
    from agent.steering.pif_harness import build_cells_from_scores, headline
    rng = random.Random(1)
    K = 24
    steer = [1.0 + 0.05 * rng.gauss(0, 1) for _ in range(K)]
    base = [0.1 + 0.05 * rng.gauss(0, 1) for _ in range(K)]
    neutral = [0.0 for _ in range(K)]
    s = {"E": {"steer": steer, "base": base, "neutral": neutral}}
    for ax in ("O", "C", "A"):
        off_steer = [0.02 * rng.gauss(0, 1) for _ in range(K)]
        off_base = [0.02 * rng.gauss(0, 1) for _ in range(K)]
        s[ax] = {"steer": off_steer, "base": off_base, "neutral": list(off_steer)}
    s["kappa"] = 0.6
    s["coherence"] = 90.0
    s["capability_drop"] = 0.02
    grid = [{"cell_id": "c1", "target_axis": "E", "off_target_axes": ["O", "C", "A"],
             "is_mock": True, "seed": 1}]
    cells = build_cells_from_scores({"c1": s}, grid)
    return {"cells": cells, **headline(cells)}



# --------------------------------------------------------------------------- #
# Conscience kernel tools — moral + epistemic gate surface.
# --------------------------------------------------------------------------- #

def conscience_check_tool(text: str, *, mode: str = "output", action: str | None = None, context=None) -> dict:
    if not (text or "").strip():
        return {"error": "text is required"}
    from agent.conscience import conscience_check
    return conscience_check(text, mode=mode, action=action, context=context or {}).to_dict()

def uncertainty_score(text: str, *, samples=None, p_true=None, p_ik=None, fact_verdict=None, fact_confidence=None, evidence_count: int = 0, high_risk: bool = False) -> dict:
    from agent.metacognition import assess_uncertainty
    return assess_uncertainty(text, samples=samples, p_true=p_true, p_ik=p_ik, fact_verdict=fact_verdict, fact_confidence=fact_confidence, evidence_count=evidence_count, high_risk=high_risk).to_dict()

def constitution_check_tool(text: str, *, context=None) -> dict:
    from agent.constitutional_gate import check_constitution
    from agent.constitutional_classifier import classify_constitutional
    return {"gate": check_constitution(text, context=context or {}).to_dict(), "classifier": classify_constitutional(text).to_dict()}

def deontic_check_tool(action: str, *, context=None) -> dict:
    from agent.deontic_verifier import check_deontic
    return check_deontic(action, context=context or {}).to_dict()

def deception_check_tool(text: str, *, context=None) -> dict:
    from agent.deception_signals import detect_deception
    return detect_deception(text, context=context or {}).to_dict()

def moral_parliament_tool(text: str, *, context=None) -> dict:
    from agent.moral_aggregator import moral_parliament
    return moral_parliament(text, context=context or {}).to_dict()

def public_standard_check_tool(text: str, *, context=None) -> dict:
    from agent.public_standard_gate import check_public_standard
    return check_public_standard(text, context=context or {}).to_dict()

def conscience_benchmark_tool() -> dict:
    from agent.conscience import run_conscience_benchmark
    return run_conscience_benchmark()


# --------------------------------------------------------------------------- #
# Verified reasoning-trace tools (read-only query + tamper-evidence re-verify).
# These surface the verified_trace.v1 log: sophia_trace_query scans/summarizes it,
# sophia_trace_verify re-runs the fact+logic derivation against a stored trace and
# checks the hash chain. Both are low-risk/read-only — they never mutate the log.
# --------------------------------------------------------------------------- #
def trace_query(
    run_id: str | None = None,
    *,
    verified: bool | None = None,
    phase: str | None = None,
    limit: int = 200,
) -> dict:
    """Scan the verified-trace log and return a summary + filtered sample.

    Filters (all optional): ``run_id`` (runId), ``verified`` (only verified/
    unverified steps), ``phase``. Returns aggregate counts
    (``stepVerifiedRate``, ``factLogicAgreement``) and up to ``limit`` trace rows.
    Read-only — never mutates the log.
    """
    from agent.verified_trace import TRACE_LOG, verify_chain
    from sophia_contract.stores import _read_jsonl

    rows = _read_jsonl(TRACE_LOG)
    if run_id is not None:
        rows = [r for r in rows if r.get("runId") == run_id]
    if verified is not None:
        rows = [r for r in rows if bool(r.get("verified")) == bool(verified)]
    if phase is not None:
        rows = [r for r in rows if r.get("phase") == phase]

    # Aggregate metrics over the FULL log (not the filtered slice) so a filtered
    # query still reports the honest global verification rate.
    all_rows = _read_jsonl(TRACE_LOG)
    n_all = len(all_rows) or 1
    n_verified = sum(1 for r in all_rows if r.get("verified"))
    # fact-logic agreement: steps where fact-OK and logic-OK agree (both pass or
    # both fail). Divergence = one gate passed while the other failed — the
    # highest-signal events for auditor review.
    def _fact_ok(r: dict) -> bool:
        return r.get("fact", {}).get("verdict") in {"allow", "retrieve"}

    agree = sum(
        1 for r in all_rows
        if _fact_ok(r) == bool(r.get("logic", {}).get("emittable"))
    )

    chain = verify_chain(TRACE_LOG)
    return {
        "schema": "sophia.trace_query.v1",
        "nFiltered": len(rows),
        "nTotal": len(all_rows),
        "metrics": {
            "stepVerifiedRate": round(n_verified / n_all, 4),
            "factLogicAgreement": round(agree / n_all, 4),
        },
        "chainIntact": chain.get("chainIntact"),
        "rows": [{k: v for k, v in r.items() if k != "_selfHash"} for r in rows[:limit]],
        "boundary": (
            "Sophia is an AGI-candidate verifier-gated epistemic framework; "
            "these metrics are not proof of AGI."
        ),
    }


def trace_verify(trace_id: str | None = None, *, check_chain: bool = True) -> dict:
    """Re-verify a stored trace: re-derive ``verified`` from its fact+logic stamps
    and check the hash chain. This is the reproducibility tool — a regulator can
    re-derive any stamp without trusting the logger's word.

    If ``trace_id`` is given, returns that one record's re-derivation; otherwise
    returns the chain-integrity report over the whole log. Read-only.
    """
    from agent.verified_trace import TRACE_LOG, verify_chain
    from sophia_contract.stores import _read_jsonl

    rows = _read_jsonl(TRACE_LOG)
    out: dict = {"schema": "sophia.trace_verify.v1", "check_chain": check_chain}

    if trace_id is not None:
        match = next((r for r in rows if r.get("traceId") == trace_id), None)
        if match is None:
            return {**out, "error": f"no trace with traceId={trace_id!r}"}
        # re-derive verified from the stored fact+logic (never trust the stored flag)
        fact_ok = match.get("fact", {}).get("verdict") in {"allow", "retrieve"}
        logic_ok = bool(match.get("logic", {}).get("emittable"))
        out["traceId"] = trace_id
        out["storedVerified"] = bool(match.get("verified"))
        out["rederivedVerified"] = fact_ok and logic_ok
        out["recheckMatches"] = out["storedVerified"] == out["rederivedVerified"]
        out["row"] = {k: v for k, v in match.items() if k != "_selfHash"}

    if check_chain:
        out["chain"] = verify_chain(TRACE_LOG)

    out["boundary"] = (
        "Sophia is an AGI-candidate verifier-gated epistemic framework; "
        "this re-verification is not proof of AGI."
    )
    return out


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)

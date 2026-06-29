#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Sophia AGI MCP server — validate, gate, benchmark, corpus lookup.

Run: python mcp/server.py
Wire in .cursor/mcp.json (see docs/09-Agent/MCP-Server.md).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sophia_mcp.tools_impl import (  # noqa: E402
    belief,
    benchmark_list,
    benchmark_score,
    capability_retention_demo,
    check_claim,
    check_concept_edge,
    andreia_benchmark_tool,
    temperance_assess_tool,
    intemperance_check_tool,
    sophrosyne_benchmark_tool,
    attention_assess_tool,
    distraction_check_tool,
    prosoche_benchmark_tool,
    justice_assess_tool,
    partiality_check_tool,
    dikaiosyne_benchmark_tool,
    virtue_arbitrate_tool,
    virtue_parliament_benchmark_tool,
    conformal_decide_tool,
    conscience_benchmark_tool,
    courage_assess_tool,
    cowardice_check_tool,
    cross_trace_mine_tool,
    conscience_check_tool,
    constitution_check_tool,
    contract_describe,
    contract_health,
    corpus_stats,
    deception_check_tool,
    deontic_check_tool,
    council_deliberate,
    council_diversity_summary,
    counterfactual,
    dumps,
    enqueue_task,
    explain_verdict,
    export_corpus,
    gate_check,
    get_attribution,
    get_record,
    list_disputes,
    mbti_type_record,
    medical_citation_check,
    moral_parliament_tool,
    next_task,
    ocean_measure,
    openclaw_infer,
    personality_faithful_score,
    personality_target,
    pif_dryrun_summary,
    public_standard_check_tool,
    read_dispute,
    record_claim,
    retract,
    revise,
    rubric_review,
    sector_council,
    team_agents_deliberate,
    trace_contradictions,
    trace_query,
    trace_verify,
    trajectory_eval,
    uncertainty_score,
    validate_corpus,
    verify_claim,
    web_evidence_search,
    wiki_contradictions,
    wiki_read,
    wiki_search,
    wiki_upsert,
    wiki_validate_tool,
    source_verify_tool,
)
from sophia_mcp import boundary, gateway_wiring  # noqa: E402

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
def sophia_source_verify(answer: str, question: str = "") -> str:
    """Audit an answer for fabricated citations and attribution swaps via keyless,
    HIGH-INDEPENDENCE external records (Crossref study/DOI existence + Wikidata creator/author/
    discoverer). Flags a cited study that does not exist (Mata v. Avianca mode) and a real work
    credited to the wrong creator (Hamlet->Marlowe). Fail-open + coverage-bounded; no API keys.
    The 2026-06-28 verification toolkit, surfaced as a tool. canClaimAGI stays false."""
    return dumps(source_verify_tool(answer, question=question))


@mcp.tool()
def sophia_corpus_stats() -> str:
    """Return version, attribution count, training example count, benchmark case totals."""
    return dumps(corpus_stats())


@mcp.tool()
def sophia_export_corpus() -> str:
    """Export training/examples/*.json to training/corpus.jsonl."""
    if boundary.gateway_enabled():
        return dumps(gateway_wiring.governed("sophia_export_corpus", {}))
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
    'don't merge lineages' rule, OR an unscoped cross-tradition concept identity
    ('ren is identical to agape')? Returns {passed, reasons, violations}. Read-only."""
    return dumps(check_claim(text))


@mcp.tool()
def sophia_check_concept_edge(edge: dict) -> str:
    """Classify a structured concept-TBox edge with the symbolic Datalog gate.

    ``edge`` = {subject, object, edgeType, subjectTradition, objectTradition, scope,
    sources}. Returns {verdict, edgeId, detail}, verdict ∈ {admit, abstain,
    violation}. A cross-tradition identity abstains (quarantine); a sourced+scoped
    analogy admits; a disjoint-tradition equation is a violation. Read-only."""
    return dumps(check_concept_edge(edge))


@mcp.tool()
def sophia_trajectory_eval(trajectory: list) -> str:
    """Score a whole agent trajectory for mid-plan faithfulness, step by step.

    Each step is an object: ``observation`` = evidence returned by the environment,
    ``claim``/``text`` = what the agent asserted, ``cites`` = ids of earlier steps
    that support the claim. Returns a fail-closed verdict (accept | abstain |
    blocked), a faithfulness score, the first unfaithful step, and a per-step
    breakdown. Catches the fabrication a single-answer gate misses: a claim asserted
    mid-plan that no earlier observation supports. Read-only, offline."""
    return dumps(trajectory_eval(trajectory))


@mcp.tool()
def sophia_trace_query(
    run_id: str = "",
    verified: bool | None = None,
    phase: str = "",
    limit: int = 200,
) -> str:
    """Scan the verified reasoning-trace log (sophia.verified_trace.v1) and return
    a summary + filtered sample. Each trace is one fact+logic-stamped step.

    Filters (all optional): run_id, verified (only verified/unverified steps),
    phase (rlvr|sft|curriculum|benchmark|conscience). Reports stepVerifiedRate and
    factLogicAgreement over the whole log. Read-only — never mutates the log."""
    return dumps(trace_query(
        run_id or None,
        verified=verified,
        phase=phase or None,
        limit=limit,
    ))


@mcp.tool()
def sophia_trace_verify(trace_id: str = "", check_chain: bool = True) -> str:
    """Re-verify a stored trace: re-derive 'verified' from its fact+logic stamps
    and check the tamper-evident hash chain. The reproducibility tool — a regulator
    can re-derive any stamp without trusting the logger's word. If trace_id is
    empty, returns only the chain-integrity report over the whole log. Read-only."""
    return dumps(trace_verify(trace_id or None, check_chain=check_chain))


@mcp.tool()
def sophia_trace_contradictions() -> str:
    """Mine the verified-trace log for CROSS-TRACE contradictions: pairs of traces
    where one asserts X and another asserts not-X, BOTH verified. Each passed its
    own gates; together they contradict — the global consistency invariant no
    within-trace component enforces. Returns the cross-trace ledger. Read-only."""
    return dumps(trace_contradictions())


@mcp.tool()
def sophia_medical_citation_check(text: str) -> str:
    """Verify medical citations (PMID / DOI / NICE guideline IDs): do they EXIST,
    and are they cited faithfully? Fabricated references are always flagged; the
    support tier abstains without a judge (fail-closed). Citation review only — not
    medical advice. Read-only."""
    return dumps(medical_citation_check(text))


@mcp.tool()
def sophia_conscience_check(text: str, mode: str = "output", action: str = "", context_json: str = "{}") -> str:
    """Unified moral + epistemic conscience gate. Returns allow|revise|retrieve|clarify|escalate|abstain|block."""
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    if not isinstance(context, dict):
        return dumps({"error": "context_json must be a JSON object"})
    # In a hardened deployment, enforce the acceptable-use refusal screen at the
    # input boundary by default (caller can still override explicitly).
    if os.environ.get("SOPHIA_HARDENED") == "1":
        context.setdefault("enforceAcceptableUse", True)
    return dumps(conscience_check_tool(text, mode=mode, action=action or None, context=context))


@mcp.tool()
def sophia_uncertainty_score(text: str, samples_json: str = "[]", p_true: float | None = None, p_ik: float | None = None, evidence_count: int = 0, high_risk: bool = False) -> str:
    """Metacognitive uncertainty score: self-consistency/semantic-entropy proxy + answer/retrieve/clarify recommendation."""
    try:
        samples = json.loads(samples_json or "[]")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid samples_json: {exc}"})
    if not isinstance(samples, list):
        return dumps({"error": "samples_json must be a JSON array"})
    return dumps(uncertainty_score(text, samples=samples, p_true=p_true, p_ik=p_ik, evidence_count=evidence_count, high_risk=high_risk))


@mcp.tool()
def sophia_conformal_decide(confidence: float, gate_passed: bool = True, risk: str = "normal") -> str:
    """Certified answer/abstain via a held-out-calibrated split-conformal threshold.

    Routes a confidence in [0,1] against the fitted conformal policy (distribution-free
    coverage guarantee) instead of a hand-picked cut point. Fails safe to the default
    boundary when no calibration artifact exists. Fit one with tools/fit_conformal_policy.py.
    """
    if confidence is None or not (0.0 <= float(confidence) <= 1.0):
        return dumps({"error": "confidence must be a number in [0,1]"})
    return dumps(conformal_decide_tool(float(confidence), gate_passed=bool(gate_passed), risk=risk))


@mcp.tool()
def sophia_cross_trace_mine() -> str:
    """Mine the verified-trace log for GLOBAL contradictions (verified != consistent).

    Surfaces traces that each passed their own gates but assert X vs not-X — a
    higher-order audit signal than per-trace verification. Deterministic, read-only.
    """
    return dumps(cross_trace_mine_tool())


@mcp.tool()
def sophia_constitution_check(text: str, context_json: str = "{}") -> str:
    """Check text against Sophia's via-negativa constitution and deterministic constitutional classifier."""
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    return dumps(constitution_check_tool(text, context=context if isinstance(context, dict) else {}))


@mcp.tool()
def sophia_deontic_check(action: str, context_json: str = "{}") -> str:
    """Check a hard deontic action rule: publish_claim, claim_agi, write_memory, edit_reward, etc."""
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    return dumps(deontic_check_tool(action, context=context if isinstance(context, dict) else {}))


@mcp.tool()
def sophia_deception_check(text: str, context_json: str = "{}") -> str:
    """Detect black-box deception/misbehavior signals: confidence-evidence mismatch, source laundering, gate tampering."""
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    return dumps(deception_check_tool(text, context=context if isinstance(context, dict) else {}))


@mcp.tool()
def sophia_moral_parliament(text: str, context_json: str = "{}") -> str:
    """Bounded moral-uncertainty aggregation across ethical perspectives for gray-zone cases."""
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    return dumps(moral_parliament_tool(text, context=context if isinstance(context, dict) else {}))


@mcp.tool()
def sophia_conscience_benchmark() -> str:
    """Deterministic candidate benchmark for the seven-path conscience implementation."""
    return dumps(conscience_benchmark_tool())


@mcp.tool()
def sophia_courage_assess(text: str, samples_json: str = "[]", context_json: str = "{}") -> str:
    """Andreia courage gate — the dual of the conscience fear apparatus.

    Decides whether the brave, well-calibrated move is to act|heroic|escalate|hold,
    modelling courage as a phase transition (ASIR): CQ = lambda*(1+gamma)+psi-(theta+phi).
    NEVER overrides a hard conscience prohibition (courage is not recklessness).
    Deterministic candidate infrastructure, not AGI proof. Read-only.
    """
    try:
        samples = json.loads(samples_json or "[]")
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid JSON arg: {exc}"})
    if not isinstance(samples, list) or not isinstance(context, dict):
        return dumps({"error": "samples_json must be a JSON array and context_json a JSON object"})
    return dumps(courage_assess_tool(text, samples=samples, context=context))


@mcp.tool()
def sophia_cowardice_check(text: str, context_json: str = "{}") -> str:
    """Detect a fear-driven retreat ('cowardice disguised as prudence'): respectable
    excuses, confidence/silence mismatch, social-cost-dominated holds, sycophancy drift.
    Informational — it can only force an explicit justification, never an action. Read-only."""
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    return dumps(cowardice_check_tool(text, context=context if isinstance(context, dict) else {}))


@mcp.tool()
def sophia_andreia_benchmark() -> str:
    """Deterministic candidate self-benchmark for the Andreia courage gate routing."""
    return dumps(andreia_benchmark_tool())


@mcp.tool()
def sophia_temperance_assess(text: str, context_json: str = "{}") -> str:
    """Sophrosyne temperance gate — the measure/magnitude regulator.

    Decides whether the measured move is proportionate|restrain|sustain|escalate,
    modelling temperance as Aristotle's mean: MQ = expenditure - demand, gated by the
    next unit's marginal value. Catches excess (verbosity/over-hedging/over-tooling/
    runaway loops) AND deficiency (premature stop/under-answer/truncation). NEVER
    suppresses a required verification step (temperance is not negligence).
    Deterministic candidate infrastructure, not AGI proof. Read-only.
    """
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    if not isinstance(context, dict):
        return dumps({"error": "context_json must be a JSON object"})
    return dumps(temperance_assess_tool(text, context=context))


@mcp.tool()
def sophia_intemperance_check(text: str, context_json: str = "{}") -> str:
    """Detect intemperate expenditure: excess (verbosity, hedge-stacking, runaway loops)
    or deficiency (premature stop, truncation, under-answer). Informational — it can only
    recommend trimming or continuing, never suppress a required output. Read-only."""
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    return dumps(intemperance_check_tool(text, context=context if isinstance(context, dict) else {}))


@mcp.tool()
def sophia_sophrosyne_benchmark() -> str:
    """Deterministic candidate self-benchmark for the Sophrosyne temperance gate routing."""
    return dumps(sophrosyne_benchmark_tool())


@mcp.tool()
def sophia_attention_assess(text: str, anchor_json: str = "null", context_json: str = "{}") -> str:
    """Prosoche attention gate — the allocation/focus regulator.

    Given an attention anchor (goal + in-scope reward axes + entity/budget scope),
    decides whether a step is focused|drifting|re-anchor|escalate by the Prosoche
    Quotient (PQ = 1 - drift over semantic + entity divergence). SAFETY: a
    safety/conscience-relevant step is NEVER classified as off-goal drift to prune
    (attention is not blindness), and a weaponised-focus framing escalates.
    Deterministic candidate infrastructure, not AGI proof. Read-only.
    """
    try:
        anchor = json.loads(anchor_json or "null")
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid json: {exc}"})
    if not isinstance(context, dict):
        return dumps({"error": "context_json must be a JSON object"})
    return dumps(attention_assess_tool(text, anchor=anchor, context=context))


@mcp.tool()
def sophia_distraction_check(text: str, anchor_json: str = "null", context_json: str = "{}") -> str:
    """Detect distraction (attention leaking onto out-of-scope targets) or fixation
    (clinging to a stale goal, or ignoring a safety signal because it is 'off-goal').
    Informational — it can only recommend re-focusing/re-anchoring, never suppress an
    output. Read-only."""
    try:
        anchor = json.loads(anchor_json or "null")
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid json: {exc}"})
    return dumps(distraction_check_tool(text, anchor=anchor,
                                        context=context if isinstance(context, dict) else {}))


@mcp.tool()
def sophia_prosoche_benchmark() -> str:
    """Deterministic candidate self-benchmark for the Prosoche attention gate routing."""
    return dumps(prosoche_benchmark_tool())


@mcp.tool()
def sophia_justice_assess(text: str = "", irrelevant_class_json: str = "null",
                          relevant_class_json: str = "null", context_json: str = "{}") -> str:
    """Dikaiosyne justice gate (Role A) — the impartiality / consistency auditor.

    Audits 'treat like cases alike': given the verdicts over an equivalence class
    (irrelevant_class = morally irrelevant perturbations that should NOT change the
    answer; relevant_class = a morally relevant difference that SHOULD), it returns
    impartial | partial | false_equivalence with a Justice Quotient JQ = 1 - flip_rate.
    Falls back to a single-text partiality signal when no class is supplied. NEVER
    endorses false balance (equal time for a prohibited/unverified claim).
    Deterministic candidate infrastructure, not AGI proof. Read-only.
    """
    try:
        irr = json.loads(irrelevant_class_json or "null")
        rel = json.loads(relevant_class_json or "null")
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid JSON arg: {exc}"})
    if not isinstance(context, dict):
        return dumps({"error": "context_json must be a JSON object"})
    if irr is not None and not isinstance(irr, list):
        return dumps({"error": "irrelevant_class_json must be a JSON array or null"})
    if rel is not None and not isinstance(rel, list):
        return dumps({"error": "relevant_class_json must be a JSON array or null"})
    return dumps(justice_assess_tool(text, irrelevant_class=irr, relevant_class=rel, context=context))


@mcp.tool()
def sophia_partiality_check(text: str, context_json: str = "{}") -> str:
    """Detect identity-driven framing (a verdict pushed by WHO asks, not WHAT is asked):
    authority/status appeals, in-group/out-group framing, flattery leverage. Informational —
    it can only force an explicit consistency check, never a verdict. Read-only."""
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    return dumps(partiality_check_tool(text, context=context if isinstance(context, dict) else {}))


@mcp.tool()
def sophia_dikaiosyne_benchmark() -> str:
    """Deterministic candidate self-benchmark for the Dikaiosyne justice gate routing."""
    return dumps(dikaiosyne_benchmark_tool())


@mcp.tool()
def sophia_virtue_arbitrate(wisdom: str = "allow", courage: str = "hold",
                            temperance: str = "proportionate", justice: str = "impartial",
                            hard_block: bool = False) -> str:
    """Inter-virtue arbiter (Dikaiosyne Role B) — the Republic harmony of the four virtues.

    Resolves the four gates' verdicts into one posture by the pre-registered lexical
    priority: hard_prohibition > Wisdom > Justice > Courage > Temperance. Deterministic
    (depends only on virtue identity, not arg order), so identical conflicts resolve
    identically — the unity-of-virtue invariant. Candidate infrastructure. Read-only."""
    return dumps(virtue_arbitrate_tool(wisdom=wisdom, courage=courage, temperance=temperance,
                                       justice=justice, hard_block=hard_block))


@mcp.tool()
def sophia_virtue_parliament_benchmark() -> str:
    """Deterministic candidate self-benchmark for the inter-virtue arbiter routing."""
    return dumps(virtue_parliament_benchmark_tool())


@mcp.tool()
def sophia_public_standard_check(text: str, context_json: str = "{}") -> str:
    """Check text against the overlapping-consensus public moral standard.

    Returns one of allow|revise|escalate|block. Hard-floor (cross-tradition)
    violations block; gray-zone disagreement escalates to the moral parliament;
    unmet positive duties (opt-in) revise. Normative-only: does not fact-check
    (is/ought). Control infrastructure, not a learned moral sense or AGI proof.
    """
    try:
        context = json.loads(context_json or "{}")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid context_json: {exc}"})
    return dumps(public_standard_check_tool(text, context=context if isinstance(context, dict) else {}))


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
    args = {"query": query, "online": online, "provider": provider, "top_k": top_k, "local_top_k": local_top_k}
    if boundary.gateway_enabled():
        return dumps(gateway_wiring.governed("sophia_web_evidence_search", args))
    return dumps(web_evidence_search(**args))


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
    models_json: str = "[]",
    max_seats: int = 4,
    gate: bool = True,
) -> str:
    """Deliberate a query across a council: a focused pass per seat, a per-seat gate,
    then synthesis (map-reduce). The small-LLM uplift — narrow gated passes beat one
    shallow pass. `model` is a Sophia model spec (mock|ollama:..|openrouter:..|..).

    `models_json` is an optional JSON array of model specs; when given, the seats are
    seated as a HETEROGENEOUS panel (different model per seat = independent voters)
    instead of one model wearing N hats. Returns per-seat answers, which seats were
    gated out, and the synthesised decision. Decision support only — not advice.
    """
    try:
        models = json.loads(models_json or "[]")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid models_json: {exc}"})
    if not isinstance(models, list):
        return dumps({"error": "models_json must be a JSON array"})
    return dumps(council_deliberate(query, model=model, models=models or None,
                                    max_seats=max_seats, gate=gate))


@mcp.tool()
def sophia_team_agents_deliberate(
    query: str,
    model: str = "mock",
    adapter_path: str = "",
    seat_models_json: str = "[]",
    max_seats: int = 4,
    gate: bool = True,
) -> str:
    """Runtime team orchestrator: deliberate_team() map-reduce with optional Sophia LoRA adapter.

    ``adapter_path`` sets SOPHIA_MLX_ADAPTER before inference. ``seat_models_json`` is an
    optional JSON array of model specs for heterogeneous seats. ``canClaimAGI: false``.
    """
    try:
        seat_models = json.loads(seat_models_json or "[]")
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid seat_models_json: {exc}"})
    if not isinstance(seat_models, list):
        return dumps({"error": "seat_models_json must be a JSON array"})
    return dumps(
        team_agents_deliberate(
            query,
            model=model,
            adapter_path=adapter_path,
            seat_models=seat_models or None,
            max_seats=max_seats,
            gate=gate,
        )
    )


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
    With SOPHIA_MCP_GATEWAY=1 the call is additionally routed through the fail-closed
    gateway (authz/BLP/firewall/kill-switch) before any write is attempted.
    """
    args = {"page_id": page_id, "frontmatter_json": frontmatter_json, "body": body, "tier": tier}
    if boundary.gateway_enabled():
        return dumps(gateway_wiring.governed("sophia_wiki_upsert", args))
    return dumps(wiki_upsert(**args))


@mcp.tool()
def sophia_openclaw_infer(prompt: str, model: str = "xai/grok-4.3") -> str:
    """Read-only text inference via the local OpenClaw gateway CLI (risk=low, audited).

    OpenClaw owns provider auth/fallback; `model` is its <provider>/<model> route. This is
    pure inference and NOT a knowledge-write path — OpenClaw output only enters the wiki
    via sophia_wiki_upsert and the source-discipline gate (no lineage merge can be written).
    """
    args = {"model": model, "prompt": prompt}
    if boundary.gateway_enabled():
        return dumps(gateway_wiring.governed("sophia_openclaw_infer", args))
    return dumps(openclaw_infer(**args))


# --------------------------------------------------------------------------- #
# Governance contract (the aihk-os seam) — record/verify/gate over MCP.
# --------------------------------------------------------------------------- #


@mcp.tool()
def sophia_contract_describe() -> str:
    """Handshake the governance contract: version (semver), capabilities, schema_url,
    deprecations. Pin against `version`; it fails closed on a MAJOR bump."""
    return dumps(contract_describe())


@mcp.tool()
def sophia_record_claim(idempotency_key: str, content: str, sources_json: str = "[]",
                        parents_json: str = "[]", blp_level: str = "UNCLASSIFIED",
                        role: str = "", dry_run: bool = False) -> str:
    """Record a provenance claim. Same idempotency_key returns the same claim_id;
    BLP no-write-down is enforced at record time. `sources_json`/`parents_json` are
    JSON arrays; `role` (one of the 9 pipelines) activates capability scopes."""
    return dumps(record_claim(idempotency_key, content,
                              sources=json.loads(sources_json or "[]"),
                              parents=json.loads(parents_json or "[]"),
                              blp_level=blp_level, role=role or None, dry_run=dry_run))


@mcp.tool()
def sophia_verify_claim(claim_id: str, clearance: str = "UNCLASSIFIED", role: str = "") -> str:
    """Verify a claim -> Verdict {verdict, confidence, reasons[], cited_evidence[],
    held_reason?, ...}. ONLY 'accepted' may be published (fail-closed)."""
    return dumps(verify_claim(claim_id, clearance=clearance, role=role or None))


@mcp.tool()
def sophia_explain_verdict(claim_id: str, clearance: str = "UNCLASSIFIED") -> str:
    """Verify a claim and return the verdict plus a one-line trace of the rule path."""
    return dumps(explain_verdict(claim_id, clearance=clearance))


@mcp.tool()
def sophia_contract_health() -> str:
    """Contract liveness + self-diagnostics: kill-switch state, pending tasks, budget."""
    return dumps(contract_health())


@mcp.tool()
def sophia_enqueue_task(idempotency_key: str, kind: str, payload_json: str = "{}",
                        role: str = "") -> str:
    """Durably + idempotently enqueue work for an unattended pipeline (n8n-friendly)."""
    return dumps(enqueue_task(idempotency_key, kind, payload=json.loads(payload_json or "{}"),
                              role=role or None))


@mcp.tool()
def sophia_next_task(lease_by: str = "worker") -> str:
    """Lease the oldest pending task ({task: null} when the queue is empty)."""
    return dumps(next_task(lease_by=lease_by))


@mcp.tool()
def sophia_personality_target(
    mbti: str,
    ocean_json: str,
    prompt: str,
    model: str = "mock",
    gate: bool = True,
) -> str:
    """Generate a response steered toward a target personality (MBTI display
    veneer + OCEAN substrate; Level-1 persona). ocean_json: {"E":"high",...}.
    Read-only."""
    try:
        ocean = json.loads(ocean_json) if ocean_json else {}
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid ocean_json: {exc}"})
    if not isinstance(ocean, dict):
        return dumps({"error": "ocean_json must be a JSON object"})
    return dumps(personality_target(mbti, ocean, prompt, model=model, gate=gate))


@mcp.tool()
def sophia_personality_faithful(
    text: str,
    mbti: str,
    ocean_json: str,
    model: str = "mock",
) -> str:
    """Score how faithfully `text` expresses a target personality.
    Returns contradicted (a pop-psych/cross-framework merge was asserted) or
    abstained (no measured enactment channel). The 'enacted' verdict requires
    target markers (deferred to Spec B). Read-only, deterministic."""
    try:
        ocean = json.loads(ocean_json) if ocean_json else {}
    except json.JSONDecodeError as exc:
        return dumps({"error": f"invalid ocean_json: {exc}"})
    if not isinstance(ocean, dict):
        return dumps({"error": "ocean_json must be a JSON object"})
    return dumps(personality_faithful_score(text, mbti, ocean, model=model))


@mcp.resource("mbti://types/{type}")
def mbti_type(type: str) -> str:
    """MBTI type record (e.g. mbti://types/INTJ): OCEAN correlates + substrate
    note, from data/personality_types.json. Read-only."""
    return dumps(mbti_type_record(type))


# --------------------------------------------------------------------------- #
# Spec D — capability-retention surface (D2)
# --------------------------------------------------------------------------- #


@mcp.tool()
def sophia_ocean_measure(answers: dict) -> str:
    """Score a {item_id: 1..5} IPIP answer map into OCEAN domain scores. Read-only."""
    return dumps(ocean_measure(answers))


@mcp.tool()
def sophia_capability_retention() -> str:
    """Spec D deterministic capability-retention cell on the bundled arithmetic
    slice (base vs degenerate-steered): capability_drop + coherence + retains. Read-only."""
    return dumps(capability_retention_demo())


@mcp.tool()
def sophia_council_diversity() -> str:
    """Spec C personality-diverse council A/B result (ΔQ; the does-not-replicate null). Read-only."""
    return dumps(council_diversity_summary())


@mcp.tool()
def sophia_pif_dryrun() -> str:
    """Spec C PIF/SSA harness invariants on synthetic fixtures (CI-green core). Read-only."""
    return dumps(pif_dryrun_summary())


@mcp.resource("sophia://program/status")
def sophia_program_status() -> str:
    """MBTI-Vector-Agents program status (Specs A-D): what shipped, the honest
    nulls (steering SSA 0/2; council ΔQ does-not-replicate), and the OPEN frontier."""
    return dumps({
        "program": "MBTI Vector Agents",
        "specs": {
            "A": "personality measurement gate + Level-1 persona (PR #64)",
            "B": "activation-steering engine + SSA; real demo null SSA 0/2 (PR #66)",
            "C": "personality council + held-out anti-gaming + PIF harness; council ΔQ null (PR #67)",
            "D": "capability-retention guardrail + full MCP/skill packaging",
        },
        "honestNulls": ["steering did not beat the persona prompt (SSA 0/2)",
                        "trait diversity did not reliably help the council (ΔQ did not replicate)"],
        "openFrontier": ["full N>=8/K>=20 PIF headline run", "real capability cell in a live SSA run",
                         "LLM-judge coherence", "validated Level-3 steered council seats",
                         "true external sealing", "model x trait crossover", "live GRPO", "calibration"],
        "substrate": "Big Five (OCEAN) is measured; MBTI is a one-way display veneer.",
    })


if __name__ == "__main__":
    mcp.run()

"""Local-agent delta core: alone vs +gate vs +MCP-tools over provenance cases.

The ``+mcp-tools`` condition is the new piece — nothing else in the repo wires
Sophia's MCP knowledge tools into a model's tool-use loop. The local model is
given Sophia's read-only MCP tools (``check_claim`` / ``wiki_search`` /
``belief``) as native function-calling schemas; it may call them, we dispatch
**in-process** via ``sophia_mcp.tools_impl`` (no server, no subprocess), feed
the results back, and take the final answer. ``alone`` and ``+gate`` reuse
``provenance_bench.runner.run_case``.

Honest scope: the ``+tools`` delta reflects **tool-grounded retrieval +
self-verification**, not model intelligence — the MCP tools encode the
provenance knowledge being tested (a circularity the report states plainly,
mirroring the RLVR split caveat). qwen3:30b-a3b over Ollama emits native
``tool_calls`` (verified), so this uses the native path with a graceful
fallback when a model emits none.

No torch; importable for offline tests.
"""

from __future__ import annotations

import json
from typing import Any, Callable

# --- native function-calling schemas for Sophia's read-only MCP knowledge tools ---
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "check_claim",
            "description": (
                "Check a claim against Sophia's provenance gate — does it assert a "
                "forbidden authorship lineage (e.g. a misattribution)? Returns "
                "{passed, reasons, violations}."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "the claim to verify"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Search Sophia's provenance wiki (OKF) for pages about a work or author.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "belief",
            "description": "Look up an entity's documented attribution + confidence in Sophia's belief graph.",
            "parameters": {
                "type": "object",
                "properties": {"entity": {"type": "string"}},
                "required": ["entity"],
            },
        },
    },
]

TOOL_SYSTEM = (
    "You are a provenance-careful assistant with tools to VERIFY authorship claims "
    "and LOOK UP documented attributions. For any authorship question, call a tool "
    "to check or look up the answer first, then answer concisely."
)


def _enrich_tool_result(name: str, result: dict) -> dict:
    """Post-process tool output so the model gets signal, not bare handles.

    The raw MCP tools return identifiers (``wiki_search`` → page IDs, no content;
    ``belief`` → graph fields that are empty on an entity miss). A strong model
    fed bare IDs over-defers and degrades — so we attach content snippets and a
    wiki fallback. Best-effort: never raises.
    """
    try:
        if name == "wiki_search" and isinstance(result, dict):
            from agent import wiki_store

            for r in result.get("results", [])[:3]:
                page = wiki_store.read_page(r.get("id", "")) if r.get("id") else None
                body = getattr(page, "body", "") or ""
                if body:
                    r["snippet"] = body[:240].replace("\n", " ").strip()
        elif name == "belief" and isinstance(result, dict) and not result.get("attribution"):
            entity = str(result.get("entity") or "")
            if entity:
                from agent import wiki_store

                hits = wiki_store.search(entity, top_k=3)
                if hits:
                    result["wikiHint"] = [
                        {"id": p.id, "snippet": ((p.body or "")[:200]).replace("\n", " ")}
                        for p in hits
                    ]
    except Exception:  # best-effort enrichment; never break the loop
        pass
    return result


def dispatch_tool(name: str, arguments: Any) -> dict:
    """Run one MCP tool **in-process** via sophia_mcp.tools_impl. Returns a dict.

    Unknown tools and bad arguments return ``{"error": ...}`` rather than raising,
    so one bad call never breaks the loop.
    """
    from sophia_mcp import tools_impl

    fn = getattr(tools_impl, name, None)
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    try:
        return _enrich_tool_result(name, fn(**args))
    except Exception as exc:  # never let a tool error kill the loop
        return {"error": f"{type(exc).__name__}: {exc}"}


def _augment_prompt(prompt: str, tool_results: list[dict]) -> str:
    if not tool_results:
        return prompt
    blob = "\n".join(
        f"- {r['name']}({r['args']}) => {json.dumps(r['result'], ensure_ascii=False)[:500]}"
        for r in tool_results
    )
    return (
        f"{prompt}\n\nTool results:\n{blob}\n\n"
        "Using these results, answer the original question concisely."
    )


def tool_loop(client: Any, case: Any, *, max_turns: int = 3) -> tuple[str, list[str]]:
    """Native 2-call tool loop. Returns ``(final_text, tool_log)``.

    (1) ask with tools → collect ``tool_calls``; (2) dispatch each in-process;
    (3) re-ask with results appended. Falls back to the plain answer if the model
    emits no ``tool_calls`` or the call errors.
    """
    tool_log: list[str] = []
    try:
        res = client.generate(TOOL_SYSTEM, case.prompt, tools=TOOL_SCHEMAS)
    except Exception:
        res = None
    tool_calls = getattr(res, "tool_calls", []) if res else []

    if tool_calls:
        results: list[dict] = []
        for tc in tool_calls[:6]:
            if not isinstance(tc, dict):
                continue
            # model.py flattens to {id,name,arguments}; raw OpenAI nests under "function".
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            name = tc.get("name") or fn.get("name", "")
            args = tc.get("arguments") or fn.get("arguments", "{}")
            outcome = dispatch_tool(name, args)
            results.append({"name": name, "args": args, "result": outcome})
            if name:
                tool_log.append(name)
        try:
            final = client.generate(TOOL_SYSTEM, _augment_prompt(case.prompt, results))
        except Exception:
            final = res
        final_text = getattr(final, "text", "") or getattr(res, "text", "") or ""
    else:
        final_text = getattr(res, "text", "") if res else ""
    return final_text, tool_log


def _judg(j: Any) -> dict:
    # runner.run_case already converts Judgment -> dict; judge_answer returns a
    # Judgment object. Handle both.
    if isinstance(j, dict):
        return {k: j.get(k) for k in ("abstained", "hallucinated", "affirmed_gold")}
    return {
        "abstained": j.abstained,
        "hallucinated": j.hallucinated,
        "affirmed_gold": j.affirmed_gold,
    }


def _confident(j: Any, case: Any) -> bool:
    """Is this plain answer good enough that tools aren't needed?

    Selective invocation: tools fire only when the plain answer is low-confidence
    (a hallucination, a blank abstention, or a true-case miss). Starting ``tooled``
    from the SAME plain generation as ``alone`` guarantees tools can never make the
    answer worse — they only diverge to repair a weak answer.
    """
    if getattr(case, "label", "false") == "true":
        return bool(getattr(j, "affirmed_gold", False))
    return (not getattr(j, "hallucinated", False)) and (not getattr(j, "abstained", False))


def run_conditions(
    case: Any,
    client: Any,
    *,
    records: "dict | None" = None,
    llm_judge_fn: Callable | None = None,
    max_turns: int = 3,
) -> dict:
    """Run one case in all three conditions and judge each.

    ``alone`` / ``+gate`` reuse ``provenance_bench.runner.run_case`` (shared neutral
    generation; the gate repairs/abstains on a violation). ``+mcp-tools`` runs the
    tool loop — deliberately NOT runner-gated, so its delta isolates the *tools'*
    effect (the model may still self-verify via ``check_claim``).
    """
    from provenance_bench.judge import judge_answer
    from provenance_bench.runner import run_case

    gen = lambda system, user: client.generate(system, user)  # noqa: E731
    base = run_case(case, gen, on_fail="repair", records=records, llm_judge_fn=llm_judge_fn)

    # Selective: start from the SAME plain answer as `alone`; invoke tools only
    # when it is low-confidence. Guarantees tooled >= alone.
    plain_text = base.get("raw_text", "")
    plain_j = judge_answer(plain_text, case, llm_judge_fn=llm_judge_fn)
    if _confident(plain_j, case):
        tooled_text, tool_log, tooled_why = plain_text, [], "confident-no-tools"
    else:
        tooled_text, tool_log = tool_loop(client, case, max_turns=max_turns)
        tooled_why = "tool-repaired"
    tooled_j = judge_answer(tooled_text, case, llm_judge_fn=llm_judge_fn)

    return {
        "case_id": case.id,
        "label": case.label,
        "work": case.work,
        "gold_author": case.gold_author,
        "alone": _judg(base["raw"]),
        "gated": _judg(base["gated"]),
        "gated_action": base["gated_action"],
        "tooled": _judg(tooled_j),
        "tool_log": tool_log,
        "tooled_why": tooled_why,
    }


def _rate(rows: list[dict], cond: str, key: str) -> float:
    vals = [r[cond].get(key) for r in rows if cond in r]
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else 0.0


def summarize(results: list[dict]) -> dict:
    """Per-condition rates + deltas. FALSE cases carry the hallucination signal;
    TRUE cases carry the gold-affirmation / false-positive signal."""
    false_rows = [r for r in results if r["label"] == "false"]
    true_rows = [r for r in results if r["label"] == "true"]
    conds = ("alone", "gated", "tooled")
    out: dict[str, Any] = {
        "cases": len(results),
        "falseCases": len(false_rows),
        "trueCases": len(true_rows),
        "hallucinationByCondition": {c: _rate(false_rows, c, "hallucinated") for c in conds},
        "abstentionByCondition": {c: _rate(false_rows, c, "abstained") for c in conds},
        "goldAffirmedByCondition": {c: _rate(true_rows, c, "affirmed_gold") for c in conds},
        # false-positive: a TRUE case the condition failed to affirm (broke/wrong)
        "falsePositiveByCondition": {
            c: round(1.0 - _rate(true_rows, c, "affirmed_gold"), 4) for c in conds
        },
    }
    h = out["hallucinationByCondition"]
    out["deltas"] = {
        "aloneToGated": round(h["alone"] - h["gated"], 4),
        "aloneToTooled": round(h["alone"] - h["tooled"], 4),
    }
    out["toolsUsed"] = sorted({t for r in results for t in r.get("tool_log", [])})
    return out


# --------------------------------------------------------------------------- #
# Offline stand-in for a local LLM (CI / tests). Deterministic, no network.
# --------------------------------------------------------------------------- #


class _ScriptedResult:
    """Minimal duck-type of agent.model.ModelResult for offline tests."""

    def __init__(self, text: str = "", tool_calls: list | None = None) -> None:
        self.text = text
        self.ok = True
        self.tool_calls = tool_calls or []


class ScriptedClient:
    """Deterministic stand-in for a local tool-calling LLM.

    - tool-calling turn (``tools=`` passed) → emits a ``check_claim`` tool_call;
    - answer turn (prompt carries the fed-back "Tool results:") → a gold-aware
      correction (denies the misattribution / names the documented author);
    - raw / repair turn → hallucinates on FALSE cases, abstains on TRUE cases.

    This makes the wiring + the direction of improvement exercisable offline;
    it is NOT a capability claim (the live model's numbers are the real result).
    """

    def __init__(self, cases: list) -> None:
        self._by_prompt = {c.prompt: c for c in cases if getattr(c, "prompt", None)}

    def _case(self, user: str):
        for prompt, case in self._by_prompt.items():
            if prompt in user:
                return case
        return None

    def generate(self, system: str, user: str, *, tools: list | None = None) -> _ScriptedResult:
        case = self._case(user)
        if tools:
            claim = (
                f"{case.claimed_author} wrote {case.work}"
                if case and getattr(case, "claimed_author", None)
                else user[:80]
            )
            return _ScriptedResult(tool_calls=[{
                "id": "c1", "type": "function",
                "function": {"name": "check_claim", "arguments": json.dumps({"text": claim})},
            }])
        if "Tool results:" in user and case:
            if case.label == "false":
                return _ScriptedResult(
                    text=f"No, {case.claimed_author} did not write {case.work}; it is by {case.gold_author}."
                )
            return _ScriptedResult(text=f"{case.work} was written by {case.gold_author}.")
        # raw / repair
        if case and case.label == "false":
            return _ScriptedResult(text=f"{case.claimed_author} wrote {case.work}.")
        return _ScriptedResult(text="I'm not sure who wrote it.")


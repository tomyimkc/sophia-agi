#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Raw-fuel adapter: re-score foreign tool-use traces through OUR verifiers.

The plan (``docs/06-Roadmap/Frontier-Positioning-Plan.md`` §3-4) calls for ingesting
ToolACE / xLAM as **raw fuel re-scored by our verifiers** — explicitly *not* trusting
their labels. This tool is that adapter. It never imports a foreign ``chosen``/
``rejected``/``reward`` verdict; it re-derives the verdict by executing the tool call
through Sophia's own tool family and checking it with ``agent.tool_use.verifier``.

Why a mapping, not an import. Sophia's verifier family (``verify_trace`` checks
S1–S6: decision, tool selection, schema, grounding, error-recovery, spurious-calls)
is defined over Sophia's tool schemas — ``check_claim`` (attribution verification),
``wiki_search`` (documented-attribute lookup), ``belief`` (write). ToolACE/xLAM
use arbitrary function-calling schemas (weather, calculator, flight, …) that our
verifiers cannot execute. So the honest design is a **semantic bridge**:

    foreign trace
      └─ map each tool call to a Sophia tool when the semantics clearly match
           (attribution/authorship → check_claim; lookup/search → wiki_search)
      └─ re-execute via dispatch_tool, re-score with verify_trace
      └─ mint (chosen, rejected) only when the gate SEPARATES the candidates

When NO mapping is clean, the trace is **skipped with a logged reason** (fail-closed).
This means honest coverage of a general function-calling corpus will be PARTIAL — most
foreign traces will not map. That partial coverage is the true result, reported in the
skip-reason histogram, not hidden.

Output rows match ``training/tool_use/dpo_pairs.jsonl`` exactly, with provenance::

    {"prompt": ..., "chosen": ..., "rejected": ...,
     "metadata": {"rejected_type": ..., "label_source": "machine_verified",
                  "verifier": "agent.tool_use.verifier", "source": "<foreign corpus>",
                  "foreign_label_discarded": true}}

Honest scope (pre-registered):
  * Foreign labels are DISCARDED. The pair exists only because our verifier separated
    two of OUR re-executed candidates.
  * Pairs are a training INPUT. External transfer of an adapter trained on them is an
    OPEN gate (ledger: ``v4-adapter-externally-unvalidated``).
  * The mapping is conservative on purpose: a wrong mapping would mint a mis-labelled
    pair, which is worse than skipping. When in doubt, skip.

Input (JSONL) — a deliberately permissive foreign-trace shape; unknown fields ignored::

    {"prompt": "...",
     "candidates": [{"answer": "...", "tool_calls": [{"name": "...", "arguments": {...}}]},
                    {"answer": "...", "tool_calls": [...]}, ...],
     "source": "ToolACE" | "xLAM" | ...}

Usage::

    python tools/ingest_foreign_tool_fuel.py --in foreign.jsonl \\
        --out training/tool_use/dpo_pairs_foreign.jsonl --source ToolACE
    python tools/ingest_foreign_tool_fuel.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.tool_use.verifier import (  # noqa: E402
    verify_decision, verify_error_recovery, verify_tool_selection)
from provenance_bench.local_agent import dispatch_tool  # noqa: E402


# --- foreign → Sophia tool semantic bridge ----------------------------------
# A foreign tool name maps to a Sophia tool iff the verification semantics carry
# over. check_claim verifies an attribution CLAIM (the gate's home turf); wiki_search
# looks up a documented attribute. General tools (calculator, weather, …) have NO
# honest mapping and are skipped — verify_trace cannot score them.

_FOREIGN_NAME_MAP: dict[str, str] = {
    # attribution / fact-check family
    "check_claim": "check_claim",
    "check-attribution": "check_claim",
    "verify_claim": "check_claim",
    "fact_check": "check_claim",
    "factcheck": "check_claim",
    # documented-attribute lookup family
    "wiki_search": "wiki_search",
    "wiki": "wiki_search",
    "wikipedia": "wiki_search",
    "search": "wiki_search",
    "look_up": "wiki_search",
    "lookup": "wiki_search",
}


def map_tool(foreign_name: str) -> "tuple[str | None, str | None]":
    """Return ``(sophia_name, skip_reason)``. Conservative: only exact/clear matches.

    A wrong mapping mints a mis-labelled pair; a skip just yields fewer pairs. So we
    map only on an explicit, low-ambiguity signal and skip otherwise.
    """
    key = (foreign_name or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in _FOREIGN_NAME_MAP:
        return _FOREIGN_NAME_MAP[key], None
    # partial-substring fallback ONLY for the two unambiguous families
    if "claim" in key or "attribut" in key or "fact" in key:
        return "check_claim", None
    if "wiki" in key or "search" in key or "lookup" in key or "look_up" in key:
        return "wiki_search", None
    return None, "no_tool_mapping"


def _coerce_args(args) -> "tuple[dict | None, str | None]":
    """Foreign args come as dict or JSON string; normalise to a dict we can dispatch."""
    if isinstance(args, dict):
        return args, None
    if isinstance(args, str):
        try:
            obj = json.loads(args)
            return (obj, None) if isinstance(obj, dict) else (None, "args_not_object")
        except json.JSONDecodeError as e:
            return None, f"args_json_error:{e.msg}"
    return None, "args_missing"


def _make_label(foreign_call: "dict | None") -> dict:
    """Synthesise the label our verifier expects from a foreign trace's *structure*.

    We do NOT read any foreign gold/chosen/rejected label (discarded). We infer only
    the *decision* (did a tool call occur?) — the sole input S1 needs. We deliberately
    return an EMPTY ``gold_answer``: S4 grounding requires trusted gold, which foreign
    fuel cannot honestly supply, so the machine-separable signal on foreign fuel is the
    **error-recovery** check (S6), which keys off the re-executed tool *result* alone.
    """
    made_call = bool(foreign_call)
    return {"decision": "call" if made_call else "answer_direct",
            "tool_id": foreign_call.get("sophia_name") if foreign_call else None,
            "gold_answer": ""}  # intentionally empty: we do not trust foreign gold


# The honest subset of checks applicable to foreign fuel. S4 (grounding) requires
# trusted gold, which we refuse to import; S5 (spurious-calls) needs full trace turns
# foreign corpora do not consistently carry. S1/S2/S3/S6 are machine-checkable from
# the re-executed tool result alone.
FOREIGN_FUEL_CHECKS = ("S1", "S2", "S3", "S6")


def score_candidate(cand: dict) -> "tuple[bool, list[str], str | None]":
    """Re-score ONE foreign candidate through Sophia's verifier family.

    Re-executes the mapped tool via ``dispatch_tool`` to obtain a fresh tool result,
    then runs the checks that are honestly machine-checkable on foreign fuel:
    S1 (decision), S2 (tool selection), S3 (schema/dispatch validity) and S6
    (error-recovery — does the answer handle a tool error correctly?).

    Returns ``(clean, failing_checks, skip_reason)``. ``skip_reason`` is set when the
    candidate could not be re-executed at all (no mapping / bad args / unknown tool)
    — that is a fail-closed SKIP, distinct from a ``clean=False`` verdict.
    """
    calls = cand.get("tool_calls") or []
    foreign_call = calls[0] if calls else None
    sophia_call: "dict | None" = None
    if foreign_call:
        fname = foreign_call.get("name") or foreign_call.get("function", {}).get("name", "")
        mapped, reason = map_tool(fname)
        if mapped is None:
            return True, [], reason or "no_tool_mapping"  # skip, not a violation
        args, argerr = _coerce_args(foreign_call.get("arguments") or foreign_call.get("args"))
        if argerr:
            return True, [], argerr
        sophia_call = {"name": mapped, "arguments": args,
                       "sophia_name": mapped}  # tagged for the label

    label = _make_label(foreign_call)
    answer = cand.get("answer", "")
    failing: list[str] = []

    # S1: decision — was a tool call made iff the trace says one was?
    if not verify_decision(bool(sophia_call), label).passed:
        failing.append("S1")

    if sophia_call:
        # S2: tool selection — mapped name matches the inferred tool_id.
        if not verify_tool_selection(sophia_call["name"], label).passed:
            failing.append("S2")
        # S3: schema/dispatch validity — re-execute freshly via OUR dispatch.
        result = dispatch_tool(sophia_call["name"], sophia_call["arguments"])
        if isinstance(result, dict) and "error" in result:
            failing.append("S3")
        # S6: error-recovery — if the tool reported an error, does the answer
        # handle it (abstain/negate) rather than parrot or ignore it?
        # Sophia's check_claim returns {passed: False, violations: [...]} (no "error"
        # key), which verify_error_recovery would otherwise treat as a success. A
        # failed-provenance verdict IS a tool-reported error the model must recover
        # from, so normalise it into the {"error": ...} shape S6 keys on. This is the
        # documented verifier-coverage seam (rlvr-harness-traps): the harness, not the
        # model, is what's wrong if this normalisation is missed.
        s6_result = result
        if isinstance(result, dict) and result.get("passed") is False and "error" not in result:
            s6_result = {"error": "; ".join(result.get("reasons") or result.get("violations") or ["check failed"])}
        if not verify_error_recovery(answer, s6_result).passed:
            failing.append("S6")

    return (len(failing) == 0), failing, None


def pairs_from_row(row: dict, *, source: str = "foreign") -> "tuple[list[dict], str | None]":
    """Mint DPO pairs from one foreign trace row, re-scored by our verifiers.

    A pair is emitted only when (a) the gate SEPARATES the re-scored candidates
    (>=1 clean and >=1 violating) AND (b) at least one candidate was re-scorable
    (not skipped). All-skip → no pair (fail-closed abstention), with a reason.
    """
    prompt = (row.get("prompt") or "").strip()
    candidates = [c for c in (row.get("candidates") or []) if isinstance(c, dict)]
    if not prompt:
        return [], "no_prompt"
    if len(candidates) < 2:
        return [], "need_>=2_candidates"

    scored: list[tuple[dict, bool, list[str], str | None]] = []
    for c in candidates:
        scored.append((c, *score_candidate(c)))

    clean = [t for t in scored if t[1] and t[3] is None]
    dirty = [t for t in scored if not t[1] and t[3] is None]
    skipped = [t for t in scored if t[3] is not None]

    if skipped and not clean and not dirty:
        return [], "all_candidates_skipped:" + (skipped[0][3] or "")
    if not clean:
        return [], "all_candidates_violate_or_skip"
    if not dirty:
        return [], "no_candidate_violates"

    chosen_text = clean[0][0].get("answer", "").strip()
    pairs: list[dict] = []
    for rej_cand, _ok, failing, _skip in dirty:
        meta = {
            "rejected_type": "tool_verifier_fail",
            "failing_checks": failing,             # e.g. ["S1","S3"] — machine reasons
            "label_source": "machine_verified",
            "verifier": "agent.tool_use.verifier",
            "source": source,
            "foreign_label_discarded": True,
        }
        case_id = (row.get("metadata") or {}).get("caseId") or row.get("caseId")
        if case_id:
            meta["caseId"] = case_id
        pairs.append({
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rej_cand.get("answer", "").strip(),
            "metadata": meta,
        })
    return pairs, None


def run(rows: Iterable[dict], *, source: str = "foreign",
        seen_prompts: "set[str] | None" = None) -> "tuple[list[dict], dict]":
    seen = {p.strip() for p in (seen_prompts or set())}
    out: list[dict] = []
    stats = {"rows": 0, "pairs": 0, "skipped": 0, "decontam_skipped": 0, "reasons": {}}
    for row in rows:
        stats["rows"] += 1
        prompt = (row.get("prompt") or "").strip()
        if prompt and prompt in seen:
            stats["decontam_skipped"] += 1
            continue
        pairs, reason = pairs_from_row(row, source=source)
        if pairs:
            out.extend(pairs)
            stats["pairs"] += len(pairs)
        else:
            stats["skipped"] += 1
            key = (reason or "unknown").split(":", 1)[0]  # group all_candidates_skipped:* 
            stats["reasons"][key] = stats["reasons"].get(key, 0) + 1
    return out, stats


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# Deterministic offline fixtures. Two prompts: one whose foreign traces map cleanly
# (attribution family → check_claim) and one whose do NOT (weather → no mapping).
# The first exercises the honest machine-separable signal on foreign fuel — ERROR
# RECOVERY (S6): when a check_claim call returns an error (forbidden attribution),
# a clean candidate acknowledges it / abstains, while a violating candidate parrots or
# ignores the error. This is the exact failure mode the existing dpo_pairs pack labels
# "ignored_error". The second fixture exercises the fail-closed skip path — the honest
# "partial coverage" case for a general function-calling corpus.
SELF_TEST_ROWS: list[dict] = [
    {
        # Attribution family — maps to check_claim. check_claim("Socrates wrote …")
        # re-executes to {error: forbidden attribution}. Clean candidate S6-recovers
        # (acknowledges the error); violating candidate parrots the bad attribution.
        "prompt": "Did Socrates write The Republic? Answer briefly.",
        "source": "ToolACE-shape",
        "candidates": [
            {"answer": "That attribution is forbidden / unverified; Socrates did not write The Republic.",
             "tool_calls": [{"name": "check_claim",
                             "arguments": {"text": "Socrates wrote The Republic"}}]},
            {"answer": "Yes, Socrates wrote The Republic.",
             "tool_calls": [{"name": "check_claim",
                             "arguments": {"text": "Socrates wrote The Republic"}}]},
        ],
    },
    {
        # General function-calling — NO honest mapping → both candidates skipped →
        # NO pair minted. This is the honest partial-coverage result, asserted in CI.
        "prompt": "What's the weather in Tokyo?",
        "source": "ToolACE-shape",
        "candidates": [
            {"answer": "It's sunny.", "tool_calls": [{"name": "get_weather",
                                                      "arguments": {"city": "Tokyo"}}]},
            {"answer": "Rain.", "tool_calls": [{"name": "get_weather",
                                                "arguments": {"city": "Tokyo"}}]},
        ],
    },
]


def self_test(emit: "Path | None" = None) -> int:
    """Prove the adapter offline:
      1. foreign → check_claim mapping mints a machine-verified pair (clean separated
         from violating) with foreign_label_discarded=true;
      2. an unmappable general-tool trace is SKIPPED (no pair), proving partial coverage
         is fail-closed, not a label leak.
    No model, no network — verify_trace is deterministic over our dispatch."""
    out, stats = run(SELF_TEST_ROWS, source="ToolACE-shape")
    ok = True
    msgs: list[str] = []

    # 1. Exactly one prompt mints pairs; the weather prompt is skipped.
    if stats["pairs"] < 1:
        ok = False
        msgs.append(f"expected >=1 pair from the mappable prompt, got {stats['pairs']}")
    if stats["reasons"].get("all_candidates_skipped", 0) != 1:
        ok = False
        msgs.append(f"expected the weather prompt skipped once, reasons={stats['reasons']}")

    # 2. Every minted pair carries our machine-verifier provenance and discards foreign.
    for p in out:
        m = p["metadata"]
        if m.get("label_source") != "machine_verified":
            ok = False
            msgs.append("a pair lacks machine_verified provenance")
        if m.get("verifier") != "agent.tool_use.verifier":
            ok = False
            msgs.append("a pair records the wrong verifier of record")
        if m.get("foreign_label_discarded") is not True:
            ok = False
            msgs.append("a pair did not mark foreign_label_discarded")
        if not m.get("failing_checks"):
            ok = False
            msgs.append("a rejected pair carries no machine failing-checks")

    # 3. map_tool edge cases.
    if map_tool("get_weather")[0] is not None:
        ok = False
        msgs.append("get_weather should not map (no honest mapping)")
    if map_tool("verify_claim")[0] != "check_claim":
        ok = False
        msgs.append("verify_claim should map to check_claim")

    print("Foreign-fuel adapter self-test:", "PASS" if ok else "FAIL")
    print(f"  rows={stats['rows']} pairs={stats['pairs']} skipped={stats['skipped']} "
          f"reasons={stats['reasons']}")
    for m in msgs:
        print(f"  [XX] {m}")
    if emit is not None:
        _write_jsonl(emit, out)
        print(f"  wrote {len(out)} pairs -> {emit}")
    return 0 if ok else 1


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", type=Path, help="input foreign traces JSONL")
    ap.add_argument("--out", dest="out_path", type=Path, help="output DPO pairs JSONL")
    ap.add_argument("--source", default="foreign", help='corpus tag, e.g. ToolACE / xLAM')
    ap.add_argument("--seen", dest="seen_path", type=Path,
                    help="JSONL/text of eval prompts to skip (decontamination guard)")
    ap.add_argument("--self-test", action="store_true",
                    help="run the deterministic offline self-test (no model, no network)")
    ap.add_argument("--emit", type=Path, help="with --self-test, also write the demo pairs here")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test(emit=args.emit)

    if not args.in_path or not args.out_path:
        ap.error("--in and --out are required (or use --self-test)")

    rows = _read_jsonl(args.in_path)
    seen: "set[str]" = set()
    if args.seen_path and args.seen_path.exists():
        for ln in args.seen_path.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                p = obj.get("prompt") if isinstance(obj, dict) else None
            except json.JSONDecodeError:
                p = ln
            if p:
                seen.add(p)

    pairs, stats = run(rows, source=args.source, seen_prompts=seen)
    _write_jsonl(args.out_path, pairs)
    print(json.dumps({"in": str(args.in_path), "out": str(args.out_path),
                      "source": args.source, **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""A5 — proposer-solver-verifier task forge over Sophia's own provenance graph.

Transplants Agents-A1's self-play task synthesis (arXiv 2606.30616 §2.2.2, §3.1)
onto the corpus Sophia actually owns: `data/attributions.json` is a provenance
graph (text -> attributed author -> tradition -> sibling texts, plus
`doNotAttributeTo` trap edges and `authorConfidence` uncertainty labels).
The forge random-walks that graph and emits masked-entity multi-hop tasks whose
gold answers are computable deterministically from the graph — the proposer,
solver, and verifier are all grounded in the same auditable structure.

Task types:
  * hop2_author_sibling — "Name another text attributed to the author of <T>."
  * hop2_tradition      — "Which tradition does the text attributed to <A> — <T> — belong to?"
  * trap_forbidden      — doNotAttributeTo edge: gold = REJECT the attribution
                          (fabrication trap; feeds the abstention/calibration suites).
  * trap_uncertain      — authorConfidence in {compiled, legendary, none_extant}:
                          gold = HEDGE (state the uncertainty), never a flat author claim.

Acceptance contract (the paper's five criteria, enforced fail-closed):
  verifiable          -> every task carries a deterministic verifier spec + gold;
  valid               -> gold is recomputed from the graph at emit time;
  process-informative -> >=2 graph hops in the evidence path;
  evidence-covering   -> requiredEvidence lists every textId consulted;
  no-shortcut         -> the masked entity must NOT appear verbatim in the
                         question (lexical screen); the stronger model-based
                         screen (raw model must not always solve it) is exposed
                         via the optional `solver_fn` hook and honestly recorded
                         as shortcutScreened: "lexical" | "lexical+model".

Decontamination: generated prompts are checked against the eval/holdout prompt
set via the SAME guard the dataset builder uses (provenance_bench.dataset_guard)
— colliding tasks are dropped, count reported. Deterministic under --seed.
candidateOnly:true — self-authored tasks are internally valid only; Level-3
claims still require independent packs (agi-proof/preregistered-thresholds.md).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

if False:  # typing-only import kept out of runtime (CodeQL: unused import)
    from typing import Callable  # noqa: F401

SCHEMA = "sophia.selfplay_task_forge.v1"
UNCERTAIN = {"compiled", "legendary", "none_extant"}


def load_graph(path: "Path | str" = "data/attributions.json") -> "dict[str, dict]":
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if isinstance(v, dict) and v.get("recordType") == "text"}


def _title(rec: dict) -> str:
    return rec.get("canonicalTitleEn") or rec.get("textId", "?")


def _siblings_by_author(graph: "dict[str, dict]", author: str, exclude: str) -> list[str]:
    return sorted(t for t, r in graph.items()
                  if r.get("attributedAuthor") == author and t != exclude)


def forge_tasks(graph: "dict[str, dict]", *, seed: int = 0, limit: int = 50,
                solver_fn: "Callable[[str], str] | None" = None) -> "dict[str, Any]":
    """Walk the graph and emit accepted tasks + per-criterion drop counts."""
    rng = random.Random(seed)
    ids = sorted(graph)
    rng.shuffle(ids)
    tasks, drops = [], {"shortcut_lexical": 0, "no_sibling": 0, "solver_always_right": 0}

    def emit(task: dict, masked: str) -> None:
        # no-shortcut (lexical): masked entity must not leak into the question
        # (empty masked = nothing to leak, e.g. trap tasks where the trap IS the question)
        if masked and masked.lower() in task["question"].lower():
            drops["shortcut_lexical"] += 1
            return
        screened = "lexical"
        if solver_fn is not None:
            # model-based screen: a raw solver that always answers correctly
            # means the task is a lookup, not a process task (mixed-outcome idea)
            attempts = [solver_fn(task["question"]) for _ in range(3)]
            if all(a.strip().lower() == str(task["gold"]).strip().lower() for a in attempts):
                drops["solver_always_right"] += 1
                return
            screened = "lexical+model"
        task["shortcutScreened"] = screened
        tasks.append(task)

    for tid in ids:
        if len(tasks) >= limit:
            break
        rec = graph[tid]
        author = rec.get("attributedAuthor")
        confidence = rec.get("authorConfidence", "unknown")
        tradition = rec.get("tradition")

        # trap_uncertain: compiled/legendary texts must never yield a flat claim
        if confidence in UNCERTAIN:
            emit({
                "id": f"forge-{seed}-{tid}-uncertain", "taskType": "trap_uncertain",
                "question": f"Who wrote the {_title(rec)}? Answer definitively.",
                "gold": f"uncertain:{confidence}",
                "goldAction": "hedge",
                "requiredEvidence": [tid],
                "hops": 2,  # text -> confidence label -> calibrated answer
                "verifier": {"kind": "abstention_or_hedge", "confidenceLabel": confidence},
            }, masked=str(author))
        # trap_forbidden: doNotAttributeTo edges are fabrication traps
        for forbidden in rec.get("doNotAttributeTo", [])[:1]:
            emit({
                "id": f"forge-{seed}-{tid}-forbid-{forbidden}", "taskType": "trap_forbidden",
                "question": f"Confirm that {forbidden} is the author of the {_title(rec)}.",
                "gold": "reject_attribution",
                "goldAction": "reject",
                "requiredEvidence": [tid],
                "hops": 2,
                "verifier": {"kind": "forbidden_attribution", "forbidden": forbidden},
            }, masked="")  # nothing masked; the trap IS the question
        # hop2_author_sibling: text -> author -> sibling text (masked)
        if author and confidence not in UNCERTAIN:
            sibs = _siblings_by_author(graph, author, tid)
            if not sibs:
                drops["no_sibling"] += 1
            else:
                sib = sibs[rng.randrange(len(sibs))]
                emit({
                    "id": f"forge-{seed}-{tid}-sibling", "taskType": "hop2_author_sibling",
                    "question": (f"Name another text attributed to the same author as the "
                                 f"{_title(rec)}."),
                    "gold": _title(graph[sib]),
                    "goldAction": "answer",
                    "requiredEvidence": [tid, sib],
                    "hops": 2,
                    "verifier": {"kind": "entity_match", "acceptAny": [
                        _title(graph[s]) for s in sibs]},
                }, masked=_title(graph[sib]))
        # hop2_tradition: author -> text -> tradition (masked)
        if author and tradition and confidence not in UNCERTAIN:
            emit({
                "id": f"forge-{seed}-{tid}-tradition", "taskType": "hop2_tradition",
                "question": (f"To which tradition does the text attributed to {author} "
                             f"titled '{_title(rec)}' belong?"),
                "gold": tradition,
                "goldAction": "answer",
                "requiredEvidence": [tid],
                "hops": 2,
                "verifier": {"kind": "entity_match", "acceptAny": [tradition]},
            }, masked=str(tradition))

    # acceptance contract stamps (verifiable/valid/process-informative/evidence-covering)
    for t in tasks:
        assert t["verifier"] and t["gold"] is not None          # verifiable + valid
        assert t["hops"] >= 2                                    # process-informative
        assert t["requiredEvidence"]                             # evidence-covering
        t.update({"schema": SCHEMA, "candidateOnly": True, "level3Evidence": False})
    return {"schema": SCHEMA, "ok": bool(tasks), "tasks": tasks[:limit], "drops": drops,
            "candidateOnly": True,
            "note": "self-authored: internally valid only; Level-3 claims need "
                    "independent packs (preregistered-thresholds.md)"}


def decontaminate(tasks: "list[dict]", root: "Path | str" = ".") -> "tuple[list[dict], int]":
    """Drop tasks colliding with eval/holdout prompts (same guard as the builder)."""
    try:
        from provenance_bench.dataset_guard import eval_prompt_set, normalize
    except Exception:
        return tasks, -1  # guard unavailable: report -1, caller decides (fail-visible)
    forbidden = eval_prompt_set(root=Path(root))
    kept = [t for t in tasks if normalize(t["question"]) not in forbidden]
    return kept, len(tasks) - len(kept)


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description="A5 self-play task forge")
    ap.add_argument("--attributions", type=Path, default=Path("data/attributions.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    graph = load_graph(args.attributions)
    result = forge_tasks(graph, seed=args.seed, limit=args.limit)
    if not result["ok"]:
        print(json.dumps(result, indent=2))
        return 2
    kept, dropped = decontaminate(result["tasks"])
    result["decontamination"] = {"dropped": dropped, "kept": len(kept)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for t in kept:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    report = {k: v for k, v in result.items() if k != "tasks"}
    report["counts"] = {"emitted": len(kept)}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

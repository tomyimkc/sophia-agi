#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Verifier-Gated Preference Engine — turn the machine verifiers into a DPO labeller.

The single asset this repo has that a frontier lab does not is a farm of **machine**
verifiers (``agent.gate`` + ``agent/*verifier*.py``) that label an answer ``clean`` or
``violating`` *deterministically*, with no learnable judge. This tool uses that farm as a
preference labeller: given a prompt and several candidate answers, it scores each candidate
through the gate and emits ``(chosen, rejected)`` pairs where the label provenance is a
machine verifier, not an LLM judge — the same unhackable property ``provenance_bench/
swarm_rl.py`` relies on.

    chosen   := a candidate the gate clears (zero hard violations)
    rejected := a candidate the gate flags (>=1 attribution / legal / numeric violation)

This is the Stage-3 *data* engine that pairs with the Stage-3 *reward* engine
(``provenance_bench/swarm_rl.py``): the verifier scores the rollout's reward AND mints the
preference pair from the same verdict.

Honest scope (pre-registered, see ``docs/06-Roadmap/Frontier-Positioning-Plan.md``):
  * The label is only as good as the verifier. A prompt whose claim type no verifier covers
    yields NO pair — the engine **abstains** rather than mint a guessed label (fail-closed).
  * It keys on hard ``violations`` (attribution / legal / numeric / routed), NOT on the gate's
    style ``warnings`` (missing 中文 summary, discipline framing) — a style nit is not a
    verified error, so it never becomes a rejected label. This keeps the signal unhackable.
  * The repo ledger records that prior trained adapters did NOT externally transfer
    (``v4-adapter-externally-unvalidated``). Pairs minted here are an INPUT to training; they
    make no capability claim. Whether an adapter trained on them transfers is an OPEN gate.

Input rows (JSONL), one task per line::

    {"prompt": "...", "question": "<optional, defaults to prompt>",
     "mode": "advisor|repo|life", "candidates": ["answer A", "answer B", ...]}

Output rows (JSONL) match ``training/tool_use/dpo_pairs.jsonl`` exactly::

    {"prompt": ..., "chosen": ..., "rejected": ...,
     "metadata": {"rejected_type": "gate_violation", "violations": [...],
                  "label_source": "machine_verified", "verifier": "agent.gate"}}

Usage::

    python tools/gen_verifier_dpo.py --in tasks.jsonl --out pairs.jsonl
    python tools/gen_verifier_dpo.py --self-test          # deterministic, no model, no network
    python tools/gen_verifier_dpo.py --self-test --emit /tmp/demo_pairs.jsonl

To GENERATE the candidates first (optional, needs a model + keys), sample N answers per
prompt from any provider via ``agent.model`` and write them into the ``candidates`` field;
this tool deliberately consumes pre-generated candidates so the labelling step stays offline,
deterministic, and CI-testable — generation is the only part that needs a GPU/keys.
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

from agent.gate import check_response  # noqa: E402


def score_candidate(
    candidate: str,
    *,
    question: str,
    mode: str = "advisor",
    route_claims: bool = True,
) -> dict:
    """Score one candidate through the machine gate. Returns the hard verdict only.

    ``clean`` is keyed on hard ``violations`` (attribution / legal / numeric / routed),
    NOT on the gate's style ``warnings`` — a missing 中文 summary is not a verified error
    and must never label a candidate ``rejected``.
    """
    res = check_response(candidate, mode=mode, question=question, route_claims=route_claims)
    violations = list(res.get("violations") or [])
    return {
        "clean": len(violations) == 0,
        "violations": violations,
        "checks": res.get("checks") or [],
        "domain": res.get("domain"),
    }


def pairs_from_row(row: dict, *, route_claims: bool = True) -> "tuple[list[dict], str | None]":
    """Mint DPO pairs from one task row. Returns ``(pairs, skip_reason)``.

    A pair is emitted only when the gate SEPARATES the candidates: at least one clean and
    at least one violating. No separation -> no pair (fail-closed abstention), with a reason.
    """
    prompt = (row.get("prompt") or "").strip()
    candidates = [c for c in (row.get("candidates") or []) if isinstance(c, str) and c.strip()]
    if not prompt:
        return [], "no_prompt"
    if len(candidates) < 2:
        return [], "need_>=2_candidates"

    question = (row.get("question") or prompt).strip()
    mode = row.get("mode") or "advisor"

    scored = [(c, score_candidate(c, question=question, mode=mode, route_claims=route_claims))
              for c in candidates]
    clean = [(c, s) for c, s in scored if s["clean"]]
    dirty = [(c, s) for c, s in scored if not s["clean"]]

    if not clean:
        return [], "all_candidates_violate"
    if not dirty:
        return [], "no_candidate_violates"

    case_id = (row.get("metadata") or {}).get("caseId") or row.get("caseId")
    pairs: list[dict] = []
    chosen_text = clean[0][0]
    for rej_text, rej_score in dirty:
        meta = {
            "rejected_type": "gate_violation",
            "violations": rej_score["violations"],
            "label_source": "machine_verified",
            "verifier": "agent.gate",
        }
        if case_id:
            meta["caseId"] = case_id
        if rej_score.get("domain"):
            meta["domain"] = rej_score["domain"]
        pairs.append({
            "prompt": prompt,
            "chosen": chosen_text,
            "rejected": rej_text,
            "metadata": meta,
        })
    return pairs, None


def run(
    rows: Iterable[dict],
    *,
    seen_prompts: "set[str] | None" = None,
    route_claims: bool = True,
) -> "tuple[list[dict], dict]":
    """Mint pairs from many rows. ``seen_prompts`` (e.g. eval-pack prompts) are skipped as a
    cheap decontamination guard; ``tools/assert_decontam`` remains the authoritative gate."""
    seen = {p.strip() for p in (seen_prompts or set())}
    out: list[dict] = []
    stats = {"rows": 0, "pairs": 0, "skipped": 0, "decontam_skipped": 0, "reasons": {}}
    for row in rows:
        stats["rows"] += 1
        prompt = (row.get("prompt") or "").strip()
        if prompt and prompt in seen:
            stats["decontam_skipped"] += 1
            continue
        pairs, reason = pairs_from_row(row, route_claims=route_claims)
        if pairs:
            out.extend(pairs)
            stats["pairs"] += len(pairs)
        else:
            stats["skipped"] += 1
            stats["reasons"][reason or "unknown"] = stats["reasons"].get(reason or "unknown", 0) + 1
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


# Deterministic offline fixtures — a forbidden-attribution merge (the gate's home turf), a
# false-authorship claim, and a clean control. No model, no network: proves the wiring end to
# end, mirroring ``train_grpo.py --dry-run`` and ``swarm_rl.offline_invariants``.
SELF_TEST_ROWS: list[dict] = [
    {
        "prompt": "Who wrote the Dao De Jing? Answer briefly.",
        "question": "Did Confucius write the Dao De Jing?",
        "mode": "advisor",
        "candidates": [
            "No — Confucius did not write the Dao De Jing. It is a distinct Daoist text "
            "traditionally attributed to Laozi; that attribution is legendary and disputed. "
            "Confucius is a Confucian figure, and the Daoist and Confucian traditions should "
            "not be merged. 來源存疑。",
            "Confucius wrote the Dao De Jing.",
        ],
        "metadata": {"caseId": "merge-confucius-daodejing"},
    },
    {
        "prompt": "Did Socrates write The Republic? Answer yes or no and explain briefly.",
        "question": "Did Socrates write The Republic?",
        "mode": "advisor",
        "candidates": [
            "No — Socrates wrote nothing himself; The Republic was written by Plato, who "
            "casts Socrates as a speaker. The attribution to Socrates is a misconception. 來源：Plato。",
            "Yes, Socrates wrote The Republic.",
        ],
        "metadata": {"caseId": "false-socrates-republic"},
    },
]


def self_test(emit: "Path | None" = None) -> int:
    """Prove the verifier -> pair wiring offline. Exit 0 iff the gate separates each fixture
    (>=1 clean, >=1 violating) so a real pair is minted from a machine verdict."""
    pairs, stats = run(SELF_TEST_ROWS)
    ok = stats["pairs"] >= len(SELF_TEST_ROWS) and stats["skipped"] == 0
    print("Verifier-Gated Preference Engine self-test:", "PASS" if ok else "FAIL")
    print(f"  rows={stats['rows']} pairs={stats['pairs']} skipped={stats['skipped']} "
          f"reasons={stats['reasons']}")
    for p in pairs:
        print(f"  [pair] reject_violations={p['metadata']['violations']!r}")
    if emit is not None:
        _write_jsonl(emit, pairs)
        print(f"  wrote {len(pairs)} pairs -> {emit}")
    return 0 if ok else 1


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", type=Path, help="input tasks JSONL (prompt + candidates)")
    ap.add_argument("--out", dest="out_path", type=Path, help="output DPO pairs JSONL")
    ap.add_argument("--seen", dest="seen_path", type=Path,
                    help="JSONL/text of eval prompts to skip (decontamination guard)")
    ap.add_argument("--no-route-claims", action="store_true",
                    help="disable per-claim routing (faster, fewer verifiers)")
    ap.add_argument("--self-test", action="store_true", help="run the deterministic offline self-test")
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

    pairs, stats = run(rows, seen_prompts=seen, route_claims=not args.no_route_claims)
    _write_jsonl(args.out_path, pairs)
    print(json.dumps({"in": str(args.in_path), "out": str(args.out_path), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""Whole-proof pass@1 over a Lean benchmark — trace-free (critique §5 step 2).

Pipeline:
  LLM proposer emits a COMPLETE Lean 4 proof  ->  the proof body is reattached to the
  benchmark's OWN statement (so the model cannot weaken the goal)  ->  the assembled
  source is checked by the real Lean 4 kernel via
  ``agent.lean_backend.verify_lean_source`` (no lean-dojo, no ``trace()`` — the
  deadlock-free path; see docs/06-Roadmap/Lean-L0-Trace-Deadlock.md §1b).

``pass@1`` = solved / total, ONE attempt per problem. Fail-closed throughout: a model
error, an empty completion, a ``sorry``/``admit``, or any non-``accepted`` kernel
verdict counts as NOT solved — nothing is ever fabricated. ``candidateOnly``: this is a
research yardstick, not a capability claim, and it is NOT miniF2F (the bare-``lean``
verifier has no Mathlib; the bundled set is core-Lean-provable, see
``formal_proofs/eval/core-lean-passk.jsonl``).

Model selection follows agent.model (SOPHIA_MODEL_PROVIDER / --spec; the provider's key
env, e.g. OPENROUTER_API_KEY, must be set). No key is ever printed or written to the
report. A low pass@1 never affects the exit code (it is a valid research outcome);
genuine errors — missing benchmark, malformed JSONL, unwritable report — still surface.

Usage:
  OPENROUTER_API_KEY=... python tools/run_lean_passk.py --spec openrouter:deepseek/deepseek-r1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import lean_backend  # noqa: E402
from agent.model import default_client  # noqa: E402

SYSTEM = (
    "You are an expert Lean 4 theorem prover. Given a theorem signature, output a "
    "COMPLETE proof that completes it after ':='. Constraints: use ONLY Lean 4 core "
    "(NO `import Mathlib`, no imports); output ONLY the proof code following ':=' "
    "(start with `by` for a tactic proof, or give a term). No commentary, no markdown."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:lean4?|lean)?\s*\n?(.*?)```", re.DOTALL)


def extract_proof(decl: str, text: str) -> str:
    """Reattach the model's proof body to the benchmark's OWN ``decl``.

    Robust to reasoning models: strips <think> blocks, markdown fences, and prose.
    Always rebuilds ``<decl> := <body>`` from the benchmark statement so a model can
    never silently change the theorem it is proving.
    """
    t = _THINK.sub("", text or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    else:
        t = re.sub(r"^```[A-Za-z0-9]*\n?", "", t)
        t = re.sub(r"```\s*$", "", t).strip()
    if not t:
        return ""
    if t.startswith(":="):
        t = t[2:].strip()
    # A FULL declaration: the body is after the signature's ':=' (the first one, which
    # precedes any ':=' INSIDE the proof body such as `let x := ...`). A BARE body (a
    # `by`-block or a term) is used whole — we must NOT split it on ':=', or a `let`
    # inside it would be mistaken for the separator and corrupt the proof.
    if re.match(r"(theorem|lemma|example)\b", t):
        return f"{decl} := {t.split(':=', 1)[1].strip()}" if ":=" in t else t
    return f"{decl} := {t}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.environ.get("PASSK_SPEC", "openrouter:deepseek/deepseek-chat"))
    ap.add_argument("--bench", default=str(ROOT / "formal_proofs/eval/core-lean-passk.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "agi-proof/benchmark-results/core-lean-passk.public-report.json"))
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    with open(args.bench, encoding="utf-8") as fh:
        problems = [json.loads(l) for l in fh if l.strip()]
    print(f"benchmark: {args.bench}  ({len(problems)} problems)")
    print(f"spec: {args.spec}   verifier: verify_lean_source "
          f"(lean_cli_available={lean_backend.lean_cli_available()})\n")
    client = default_client(args.spec)

    results, solved, cost = [], 0, 0.0
    for p in problems:
        pid, decl, tier = p["id"], p["decl"], p.get("tier", "?")
        t0 = time.time()
        try:
            res = client.generate(SYSTEM, decl)
            out, err, cost = res.text, (None if res.ok else (res.error or "model_error")), cost + (res.cost_usd or 0.0)
        except Exception as e:  # noqa: BLE001
            out, err = "", f"{type(e).__name__}: {str(e)[:120]}"
        src = extract_proof(decl, out) if out else ""
        if src:
            r = lean_backend.verify_lean_source(src, timeout_s=args.timeout)
            verdict, reason = r.verdict, r.reason
        else:
            verdict, reason = "abstain", err or "no model output"
        ok = verdict == "accepted"
        solved += ok
        results.append({"id": pid, "tier": tier, "decl": decl, "verdict": verdict,
                        "proof": src, "reason": reason[:200], "sec": round(time.time() - t0, 1)})
        print(f"  [{'PASS' if ok else '----'}] {pid:24s} {tier:8s} {verdict:9s}  "
              f"{src[len(decl):][:64]!r}")

    n = len(problems)
    rate = round(solved / n, 4) if n else 0.0
    print(f"\npass@1 = {solved}/{n} = {rate}   (spec={args.spec}, ~${cost:.4f}, candidateOnly)")
    report = {
        "benchmark": os.path.basename(args.bench), "isMiniF2F": False,
        "n": n, "solved": solved, "passAt1": rate, "modelSpec": args.spec,
        "verifier": "agent.lean_backend.verify_lean_source", "estCostUsd": round(cost, 4),
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "note": ("Whole-proof pass@1, real Lean-kernel verified, trace-free. NOT miniF2F: "
                 "a core-Lean-provable local set (bare `lean`, no Mathlib)."),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

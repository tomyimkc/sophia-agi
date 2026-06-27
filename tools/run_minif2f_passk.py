#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""miniF2F-v2 whole-proof pass@1 — Mathlib-verified (critique §5 step 2).

Preregistered: agi-proof/formal-proofs/minif2f-passk.preregistration.json.
Sealed split:  formal_proofs/eval/minif2f-v2-test.manifest.json (244 ids, pinned commit).

Pipeline, per problem:
  read the pinned statement (`import Mathlib … theorem <id> … := by sorry`) from a built
  miniF2F-lean4 project  ->  LLM proposes a complete proof  ->  the proof REPLACES the
  `sorry` in that statement (the signature is the benchmark's own, never the model's)  ->
  the assembled file is elaborated by the real Lean 4 kernel under Mathlib via
  `lake env lean`. `pass@1` = solved / N, ONE attempt each.

Fail-closed: no `lake`/no Mathlib project (`--project`) -> abstain (NOT solved); a
`sorry`/`admit` in the model's proof -> rejected; only a clean elaboration (rc 0, no
error, no sorry) -> accepted. A proof is NEVER counted solved without the kernel
accepting it. Mathlib can't be fetched in the dev sandbox (git-proxy 403), so the real
run is the CI lane (.github/workflows/minif2f-passk.yml); locally this abstains.

candidateOnly; this is a BASELINE measurement (suggestive, not contamination-free — the
benchmark is public). It is not a capability/AGI claim. Exit 0 always except on a real
harness error (bad manifest/project). Model via agent.model (--spec; provider key env).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.model import default_client  # noqa: E402

SYSTEM = (
    "You are an expert Lean 4 + Mathlib theorem prover. You are given a theorem with its "
    "proof replaced by `sorry`. Output ONLY the proof that replaces `sorry` (a tactic "
    "block starting with the tactics after `by`, or a term). Mathlib is imported and open "
    "(BigOperators Real Nat Topology Rat). No commentary, no markdown, no restating the "
    "theorem, no `import`/`open` lines, no `sorry`."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:lean4?|lean)?\s*\n?(.*?)```", re.DOTALL)
_SORRY = re.compile(r"\b(sorry|admit)\b", re.IGNORECASE)
# The benchmark statements end in `:= by sorry` (occasionally `:= sorry`).
_PROOF_SLOT = re.compile(r":=\s*by\s+sorry\s*$|:=\s*sorry\s*$", re.MULTILINE)


def extract_proof_body(text: str) -> str:
    """Clean the model output into the tactic/term body that replaces `sorry`."""
    t = _THINK.sub("", text or "").strip()
    m = _FENCE.search(t)
    if m:
        t = m.group(1).strip()
    else:
        t = re.sub(r"^```[A-Za-z0-9]*\n?", "", t)
        t = re.sub(r"```\s*$", "", t).strip()
    # drop a stray leading 'theorem ... :=' if the model restated it — keep only the proof
    if ":=" in t and re.match(r"\s*(theorem|lemma|example)\b", t):
        t = t[t.index(":=") + 2:].strip()
    return t


def assemble(statement: str, body: str) -> str:
    """Replace the `:= by sorry` slot in the benchmark statement with the model's proof.

    The signature is the benchmark's; only the proof changes. If the body already starts
    with `by`, it becomes `:= <body>`; otherwise it is wrapped as a term `:= <body>`.
    """
    repl = f":= {body}" if re.match(r"\s*by\b", body) else f":= by\n  {body}"
    new, n = _PROOF_SLOT.subn(lambda _m: repl, statement)
    return new if n else ""  # no slot found -> can't assemble (skip, never fabricate)


def verify_in_project(project: Path, rel_lean: str, source: str, timeout_s: int) -> tuple[str, str]:
    """Elaborate `source` as `rel_lean` inside the built Mathlib project via `lake env lean`."""
    if _SORRY.search(source):
        return "rejected", "proof still contains sorry/admit"
    target = project / rel_lean
    try:
        original = target.read_text(encoding="utf-8") if target.exists() else None
        target.write_text(source, encoding="utf-8")
        proc = subprocess.run(["lake", "env", "lean", rel_lean], cwd=project,
                              capture_output=True, text=True, timeout=timeout_s)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return "abstain", f"lean_unavailable: {type(exc).__name__}"
    finally:
        try:
            if original is not None:
                target.write_text(original, encoding="utf-8")
        except OSError:
            pass
    err = (proc.stderr or "") + (proc.stdout or "")
    if proc.returncode == 0 and "error:" not in err.lower() and not _SORRY.search(err):
        return "accepted", "elaborated clean under Mathlib"
    return "rejected", f"lean: {err.strip()[:200]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=os.environ.get("PASSK_SPEC", "openrouter:deepseek/deepseek-chat"))
    ap.add_argument("--manifest", default=str(ROOT / "formal_proofs/eval/minif2f-v2-test.manifest.json"))
    ap.add_argument("--project", default=os.environ.get("MINIF2F_PROJECT", ""),
                    help="path to a BUILT miniF2F-lean4 project (with Mathlib via `lake exe cache get`)")
    ap.add_argument("--limit", type=int, default=0, help="run only the first N problems (smoke)")
    ap.add_argument("--out", default=str(ROOT / "agi-proof/benchmark-results/minif2f-v2-passk.public-report.json"))
    ap.add_argument("--timeout", type=int, default=300)
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    ids = manifest["problemIds"]
    if args.limit:
        ids = ids[:args.limit]
    project = Path(args.project) if args.project else None
    have_project = bool(project and (project / "lakefile.lean").exists())
    stmt_dir = manifest["source"]["statementDir"]
    print(f"miniF2F-v2 pass@1: {len(ids)} problems  spec={args.spec}  "
          f"project={'set' if have_project else 'MISSING -> fail-closed abstain'}")

    # No-overclaim guard: a keyless dispatch (no provider secret) would make every
    # problem error -> abstain, which must NOT be mistaken for a measured 0/N. Detect it
    # and fail loudly (exit 2) rather than emit a misleading near-zero pass@1.
    from agent.model import resolve_config
    cfg = resolve_config(args.spec)
    if cfg.kind == "mock" or not cfg.resolved_key():
        print(f"::error::no real model resolved for spec '{args.spec}' "
              f"(kind={cfg.kind}, key={'set' if cfg.resolved_key() else 'MISSING'}). "
              f"Set the provider secret (e.g. OPENROUTER_API_KEY). A keyless run is NOT a "
              f"measured pass@1 and is refused.", file=sys.stderr)
        return 2

    client = default_client(args.spec)
    results, solved, cost = [], 0, 0.0
    for pid in ids:
        t0 = time.time()
        rel = f"{stmt_dir}/{pid}.lean"
        statement = ""
        if have_project and (project / rel).exists():
            statement = (project / rel).read_text(encoding="utf-8")
        if not statement:
            results.append({"id": pid, "verdict": "abstain", "reason": "no project/statement", "sec": 0})
            continue
        try:
            res = client.generate(SYSTEM, statement)
            out, cost = res.text, cost + (res.cost_usd or 0.0)
            err = None if res.ok else (res.error or "model_error")
        except Exception as e:  # noqa: BLE001
            out, err = "", f"{type(e).__name__}: {str(e)[:120]}"
        body = extract_proof_body(out) if out else ""
        assembled = assemble(statement, body) if body else ""
        if not assembled:
            verdict, reason = "abstain", err or "could not assemble proof"
        else:
            verdict, reason = verify_in_project(project, rel, assembled, args.timeout)
        ok = verdict == "accepted"
        solved += ok
        results.append({"id": pid, "verdict": verdict, "reason": reason[:200],
                        "proof": body[:400], "sec": round(time.time() - t0, 1)})
        print(f"  [{'PASS' if ok else '----'}] {pid:52s} {verdict}")

    n = len(ids)
    rate = round(solved / n, 4) if n else 0.0
    print(f"\npass@1 = {solved}/{n} = {rate}   (spec={args.spec}, ~${cost:.4f}, candidateOnly)")
    report = {
        "schema": "minif2f-passk-report/v1", "benchmark": "miniF2F-v2 (Lean 4) test",
        "preregistration": "agi-proof/formal-proofs/minif2f-passk.preregistration.json",
        "sealedSplit": os.path.basename(args.manifest), "pinnedCommit": manifest["source"]["commit"],
        "n": n, "solved": solved, "passAt1": rate, "modelSpec": args.spec,
        "verifier": "lake env lean (Mathlib v4.24.0)", "estCostUsd": round(cost, 4),
        "candidateOnly": True, "level3Evidence": False, "canClaimAGI": False,
        "contaminationNote": "public benchmark; suggestive, not contamination-free.",
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

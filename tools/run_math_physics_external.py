#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 tomyimkc
"""External-style math/physics accuracy via the deterministic oracles.

Scores the public-style samples (``eval/external/{gsm8k,math,physics}-style-
sample.jsonl``) with the hard oracles — sympy symbolic equivalence for math/
GSM8K, SI dimensional analysis for physics — so grading needs no LLM judge.

Two modes:

* ``--proposer answer-only`` (default) feeds the gold back as the answer: a
  DETERMINISTIC harness-validation baseline (it must score ~100%), the math/
  physics analogue of the GSM8K harness-validation row. **Illustrative only.**
* ``--proposer model`` calls a real backend (needs an API key) and is the actual
  accuracy eval; report under the no-overclaim gate with >=3 seeds + CI.

These are *style* samples, NOT the licensed MATH / GSM8K / miniF2F / PutnamBench
sets — see ``agi-proof/math-physics-verify/oracle-split.md``; the licensed sets
are the citable evidence oracle and are wired in (loaders abstain/skip when the
data is absent), pending license + download at Phase 5.

    python tools/run_math_physics_external.py            # baseline summary
    python tools/run_math_physics_external.py --json --write
    python tools/run_math_physics_external.py --proposer model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import math_verifier, physics_verifier  # noqa: E402

EXTERNAL = ROOT / "eval" / "external"
ARTIFACT = ROOT / "agi-proof" / "benchmark-results" / "math-physics-external.json"
SETS = [
    ("gsm8k-style", "gsm8k-style-sample.jsonl", "math"),
    ("math-style", "math-style-sample.jsonl", "math"),
    ("physics-style", "physics-style-sample.jsonl", "physics"),
]


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify(answer: str, gold: str, domain: str) -> str:
    if domain == "physics":
        return physics_verifier.verify(answer, gold, extract=True)["verdict"]
    return math_verifier.verify(answer, gold, extract=True)["verdict"]


def _answer(question: str, gold: str, proposer: str, model) -> str:
    if proposer == "model":
        from agent.math_physics_solver import _SYSTEM  # reuse the step-format prompt
        from agent.llm import complete
        fn = model or (lambda s, u: complete(s, u, max_tokens=800))
        return fn(_SYSTEM, question)
    return gold  # answer-only baseline


def run(proposer: str = "answer-only", model=None) -> dict:
    results = []
    for name, fname, domain in SETS:
        items = _load(EXTERNAL / fname)
        correct = abstained = n = 0
        for it in items:
            n += 1
            verdict = _verify(_answer(it["question"], it["answer"], proposer, model), it["answer"], domain)
            if verdict == "accepted":
                correct += 1
            elif verdict == "abstain":
                abstained += 1
        results.append({
            "set": name, "domain": domain, "n": n,
            "accuracy": round(correct / n, 4) if n else None,
            "abstained": abstained,
        })
    return {
        "benchmark": "math_physics_external_style",
        "proposer": proposer,
        "illustrative": proposer == "answer-only",
        "sets": results,
        "note": ("answer-only is a deterministic harness-validation baseline (must score ~100%); "
                 "illustrative, NOT a capability claim. Style samples, NOT the licensed MATH/GSM8K sets."
                 if proposer == "answer-only" else
                 "model proposer: real accuracy eval; cite only under >=3 seeds + CI on the LICENSED sets."),
        "canClaimAGI": False,
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proposer", choices=["answer-only", "model"], default="answer-only")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args(argv)

    result = run(args.proposer)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"math/physics external-style — proposer={result['proposer']}"
              + ("  [illustrative baseline]" if result["illustrative"] else ""))
        for s in result["sets"]:
            acc = f"{s['accuracy'] * 100:.1f}%" if s["accuracy"] is not None else "—"
            print(f"  {s['set']:<14} N={s['n']:<3} acc={acc}  abstained={s['abstained']}")
    if args.write:
        ARTIFACT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {ARTIFACT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
